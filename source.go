package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"
)

type Frame struct {
	Seq     uint64
	CamID   string
	TrackID int
	JPEG    []byte
	Reason  string
	Score   float64
	Tracks  string // JSON list of {id,x,y,w,h} for confirmed moving objects
}

// runSource launches ffmpeg (decodes RTSP/file -> MJPEG on stdout) and emits frames.
// RTSP streams reconnect automatically; files end (or loop with -loop).
func runSource(ctx context.Context, g *Global, c *CamConfig, out chan<- Frame) {
	var seq uint64
	isRTSP := strings.HasPrefix(c.Src, "rtsp://") || strings.HasPrefix(c.Src, "rtsps://")
	for ctx.Err() == nil {
		err := runFFmpeg(ctx, g, c, isRTSP, &seq, out)
		if ctx.Err() != nil {
			return
		}
		if !isRTSP {
			if err != nil {
				log.Printf("[%s] ffmpeg: %v", c.ID, err)
			}
			return
		}
		log.Printf("[%s] stream dropped (%v), reconnecting in 3s", c.ID, err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(3 * time.Second):
		}
	}
}

func runFFmpeg(ctx context.Context, g *Global, c *CamConfig, isRTSP bool, seq *uint64, out chan<- Frame) error {
	args := []string{"-hide_banner", "-loglevel", "error"}
	if isRTSP {
		args = append(args, "-rtsp_transport", "tcp", "-timeout", "5000000")
	} else {
		if g.Realtime {
			args = append(args, "-re")
		}
		if c.Loop {
			args = append(args, "-stream_loop", "-1")
		}
	}
	args = append(args, "-i", c.Src, "-an",
		"-vf", fmt.Sprintf("fps=%d,scale=%d:-2", c.FPS, c.Width),
		"-pix_fmt", "yuvj420p", "-q:v", "4",
		"-f", "image2pipe", "-c:v", "mjpeg", "pipe:1")

	cmd := exec.CommandContext(ctx, g.FFmpeg, args...)
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cannot start ffmpeg (is it installed?): %w", err)
	}
	rerr := readJPEGs(stdout, func(j []byte) bool {
		*seq++
		select {
		case out <- Frame{Seq: *seq, CamID: c.ID, JPEG: j}:
			return true
		case <-ctx.Done():
			return false
		}
	})
	_ = cmd.Wait()
	if rerr == io.EOF {
		rerr = nil
	}
	return rerr
}

// readJPEGs splits a concatenated MJPEG stream into single JPEGs (FFD8 ... FFD9).
func readJPEGs(r io.Reader, emit func([]byte) bool) error {
	br := bufio.NewReaderSize(r, 1<<20)
	var buf []byte
	in := false
	var prev byte
	for {
		b, err := br.ReadByte()
		if err != nil {
			return err
		}
		if !in {
			if prev == 0xFF && b == 0xD8 {
				buf = append(buf[:0], 0xFF, 0xD8)
				in = true
			}
		} else {
			buf = append(buf, b)
			if prev == 0xFF && b == 0xD9 {
				frame := make([]byte, len(buf))
				copy(frame, buf)
				in = false
				prev = 0
				if !emit(frame) {
					return nil
				}
				continue
			}
		}
		prev = b
	}
}
