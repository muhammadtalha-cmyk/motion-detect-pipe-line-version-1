package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
)

//go:embed web/index.html
var indexHTML []byte

// Snapshot = numbers shared by all cameras; CamSnap = this camera's counters.
type Snapshot struct {
	Processed uint64 `json:"processed"`
	Overflow  uint64 `json:"overflow"`
	Depth     int    `json:"depth"`
}

type CamSnap struct {
	In      uint64 `json:"in"`
	Passed  uint64 `json:"passed"`
	Dropped uint64 `json:"dropped"`
}

type Event struct {
	Cam    string     `json:"cam"`
	Seq    uint64     `json:"seq"`
	JPEG   []byte     `json:"jpg"` // base64 in JSON
	Boxes  []Box      `json:"boxes"`
	Tracks []TrackBox `json:"tracks"`
	Motion bool       `json:"motion"`
	Sent   bool       `json:"sent"`
	State  string     `json:"state"`
	Reason string     `json:"reason"`
	Score  float64    `json:"score"`
	Theta  float64    `json:"theta"`
	CS     CamSnap    `json:"cs"`
	Stats  Snapshot   `json:"stats"`
	Rec    bool       `json:"rec"`
}

type RecogEvent struct {
	Type       string  `json:"type"` // always "recog"
	Cam        string  `json:"cam"`
	ID         int     `json:"id"`
	Seq        uint64  `json:"seq"`
	Status     string  `json:"status"`
	Name       string  `json:"name"`
	Similarity float64 `json:"similarity"`
	Closest    string  `json:"closest"`
	T          int64   `json:"t"`
}

type Hub struct {
	mu     sync.Mutex
	subs   map[chan []byte]struct{}
	recogs [][]byte
}

func newHub() *Hub {
	return &Hub{
		subs:   map[chan []byte]struct{}{},
		recogs: make([][]byte, 0, 100),
	}
}

func (h *Hub) Subscribe() chan []byte {
	ch := make(chan []byte, 4)
	h.mu.Lock()
	h.subs[ch] = struct{}{}
	h.mu.Unlock()
	return ch
}
func (h *Hub) Unsubscribe(ch chan []byte) {
	h.mu.Lock()
	delete(h.subs, ch)
	h.mu.Unlock()
}
func (h *Hub) HasSubs() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.subs) > 0
}
func (h *Hub) Broadcast(b []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.subs {
		select {
		case ch <- b:
		default: // slow browser: skip this frame for it
		}
	}
}

// PublishRecog appends the recognition event (up to 100) then broadcasts it outside the lock.
func (h *Hub) PublishRecog(b []byte) {
	h.mu.Lock()
	h.recogs = append(h.recogs, b)
	if len(h.recogs) > 100 {
		h.recogs = h.recogs[len(h.recogs)-100:]
	}
	h.mu.Unlock()
	h.Broadcast(b)
}

func (h *Hub) GetRecogs() [][]byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([][]byte, len(h.recogs))
	copy(out, h.recogs)
	return out
}

func serveWeb(addr string, hub *Hub, store *ShotStore) {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("/bestshots", func(w http.ResponseWriter, r *http.Request) {
		store.sentLogMu.Lock()
		log2 := make([]*SentBestShot, len(store.sentLog))
		copy(log2, store.sentLog)
		store.sentLogMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(log2)
	})
	mux.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		fl, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", 500)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		ch := hub.Subscribe()
		defer hub.Unsubscribe(ch)

		// Replay stored recognition events
		for _, b := range hub.GetRecogs() {
			fmt.Fprintf(w, "data: %s\n\n", b)
		}
		fl.Flush()

		for {
			select {
			case <-r.Context().Done():
				return
			case b := <-ch:
				fmt.Fprintf(w, "data: %s\n\n", b)
				fl.Flush()
			}
		}
	})
	log.Printf("web UI: http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
