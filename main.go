// motiongate: many RTSP/video sources -> per-camera motion gate (frame k+1 vs k-1) -> one queue -> YOLO.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
)

// CamConfig is per camera. In cameras.json any field you leave out falls back to the command-line value.
type CamConfig struct {
	ID        string  `json:"id"`
	Src       string  `json:"src"`
	FPS       int     `json:"fps"`
	Width     int     `json:"width"`
	Tau       int     `json:"tau"`
	Theta     float64 `json:"theta"`
	Step      int     `json:"step"`
	Block     int     `json:"block"`
	BlockFrac float64 `json:"blockfrac"`
	Hold      float64 `json:"hold"` // seconds to keep sending after motion stops
	Loop      bool    `json:"loop"`
	IOU       float64 `json:"iou"` // tracker: min box overlap to count as the same object
}

type Global struct {
	Addr, QueueKind, QueueURL, QName string
	QueueCap, YoloMS                 int
	SaveDir, FFmpeg                  string
	Realtime                         bool
	MPURL                            string
	MPEvery                          int
	KeepStill                        bool
	FaceOnly                         bool
	MinShot                          float64
	BestEvery                        int // seconds: how often to force-send best shot for long-duration tracks
	MP                               *PersonClient
	RecURL                           string
	Rec                              *RecClient
}

type Stats struct{ Processed, Overflow atomic.Uint64 }

func loadCameras(path string, base CamConfig) ([]CamConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f struct {
		Defaults json.RawMessage   `json:"defaults"`
		Cameras  []json.RawMessage `json:"cameras"`
	}
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(f.Defaults) > 0 {
		if err := json.Unmarshal(f.Defaults, &base); err != nil {
			return nil, fmt.Errorf("defaults: %w", err)
		}
	}
	var cams []CamConfig
	for i, raw := range f.Cameras {
		c := base
		if err := json.Unmarshal(raw, &c); err != nil {
			return nil, fmt.Errorf("camera %d: %w", i+1, err)
		}
		if c.ID == "" {
			c.ID = fmt.Sprintf("cam%d", i+1)
		}
		cams = append(cams, c)
	}
	return cams, nil
}

func main() {
	var g Global
	var base CamConfig
	var cfgPath string
	flag.StringVar(&cfgPath, "config", "", "cameras.json with many cameras (otherwise use -src for one)")
	flag.StringVar(&base.Src, "src", "", "single camera: video file or rtsp:// URL")
	flag.StringVar(&base.ID, "cam", "cam1", "single camera: id")
	flag.IntVar(&base.FPS, "fps", 15, "frames per second to read")
	flag.IntVar(&base.Width, "width", 640, "frame width in px (this is also what YOLO receives)")
	flag.IntVar(&base.Tau, "tau", 25, "pixel counts as changed if |diff| > tau (0-255)")
	flag.Float64Var(&base.Theta, "theta", 0.01, "motion if fraction of changed pixels > theta (0.01 = 1%)")
	flag.IntVar(&base.Step, "step", 2, "compare every Nth pixel (speed)")
	flag.IntVar(&base.Block, "block", 8, "box grid size (in compared pixels)")
	flag.Float64Var(&base.BlockFrac, "blockfrac", 0.10, "a block is 'moving' if this fraction of it changed")
	flag.Float64Var(&base.Hold, "hold", 2.0, "keep sending frames this many seconds after motion stops")
	flag.Float64Var(&base.IOU, "iou", 0.15, "tracker: min overlap (0-1) to keep the same ID")
	flag.BoolVar(&base.Loop, "loop", false, "loop video files forever")
	flag.BoolVar(&g.Realtime, "realtime", true, "play files at real speed (like a camera)")
	flag.StringVar(&g.Addr, "addr", ":8080", "web UI address")
	flag.StringVar(&g.QueueKind, "queue", "mem", "queue backend: mem | rabbit")
	flag.StringVar(&g.QueueURL, "amqp", "amqp://guest:guest@localhost:5672/", "RabbitMQ URL")
	flag.StringVar(&g.QName, "qname", "frames", "RabbitMQ queue name")
	flag.IntVar(&g.QueueCap, "qcap", 500, "in-memory queue capacity (frames)")
	flag.IntVar(&g.YoloMS, "yolo-ms", 50, "fake YOLO time per frame (mem queue only)")
	flag.StringVar(&g.SaveDir, "save", "", "mem queue worker saves received frames here (optional)")
	flag.StringVar(&g.MPURL, "mp", "", "MediaPipe person service URL, e.g. http://localhost:8765 (enables person mode)")
	flag.IntVar(&g.MPEvery, "mpevery", 1, "person mode: run MediaPipe on every Nth open-gate frame (1 = every frame)")
	flag.BoolVar(&g.KeepStill, "keepstill", false, "person mode: also keep boxing people who are not moving (sends frames continuously)")
	flag.BoolVar(&g.FaceOnly, "face", false, "person mode: only box/track/queue people whose FACE is visible (needs -mp)")
	flag.Float64Var(&g.MinShot, "minshot", 0.20, "minimum best-shot score to send (0-1)")
	flag.IntVar(&g.BestEvery, "bestevery", 10, "force-send best shot every N seconds for long-lasting tracks")
	flag.StringVar(&g.RecURL, "rec", "", "recognition service URL, e.g. http://localhost:8766")
	flag.StringVar(&g.FFmpeg, "ffmpeg", "ffmpeg", "ffmpeg binary")
	flag.Parse()

	var cams []CamConfig
	if cfgPath != "" {
		var err error
		if cams, err = loadCameras(cfgPath, base); err != nil {
			log.Fatal(err)
		}
	} else if base.Src != "" {
		cams = []CamConfig{base}
	}
	if len(cams) == 0 {
		fmt.Fprintln(os.Stderr, "give -src <video|rtsp://...> or -config cameras.json")
		flag.Usage()
		os.Exit(2)
	}
	seen := map[string]bool{}
	for _, c := range cams {
		if c.Src == "" || c.FPS < 1 || c.Step < 1 || c.Block < 1 || c.Width < 16 {
			log.Fatalf("camera %q: bad or missing settings (src/fps/step/block/width)", c.ID)
		}
		if seen[c.ID] {
			log.Fatalf("duplicate camera id %q", c.ID)
		}
		seen[c.ID] = true
	}

	if g.MPURL != "" {
		g.MP = NewPersonClient(g.MPURL)
		if err := g.MP.Health(); err != nil {
			log.Fatalf("MediaPipe service not reachable at %s (%v). Start it first: python3 mediapipe_service.py", g.MPURL, err)
		}
		log.Println("person mode: MediaPipe on, only people are boxed, tracked and queued")
	}
	if g.FaceOnly && g.MP == nil {
		log.Fatal("-face needs -mp")
	}
	if g.FaceOnly {
		log.Println("face mode: people without a visible face are ignored")
	}

	if g.RecURL != "" {
		g.Rec = NewRecClient(g.RecURL)
		if err := g.Rec.Health(); err != nil {
			log.Fatalf("recognition service not reachable at %s: %v. Start it first: python recognizer_service.py", g.RecURL, err)
		}
		log.Println("recognition on")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	hub := newHub()
	store := newShotStore()
	go serveWeb(g.Addr, hub, store)

	var stats Stats
	var q Queue
	switch g.QueueKind {
	case "mem":
		mq := newMemQueue(g.QueueCap, &stats)
		q = mq
		go runWorker(ctx, mq, &g, &stats, hub, store)
	case "rabbit":
		var err error
		if q, err = newRabbitQueue(g.QueueURL, g.QName); err != nil {
			log.Fatal(err)
		}
	default:
		log.Fatal("unknown -queue, use mem or rabbit")
	}
	defer q.Close()

	var wg sync.WaitGroup
	for _, c := range cams {
		wg.Add(1)
		go func(c CamConfig) {
			defer wg.Done()
			runCamera(ctx, &g, c, q, hub, &stats, store)
		}(c)
	}
	log.Printf("%d camera(s) running", len(cams))
	go func() {
		wg.Wait()
		log.Println("all sources finished. Web UI stays up; press Ctrl+C to quit.")
	}()
	<-ctx.Done()
}
