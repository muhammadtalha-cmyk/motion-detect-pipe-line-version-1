package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// runWorker is a stand-in for YOLO: it pulls frames from the queue at its own pace.
// If recognition is enabled (-rec), it calls the recognition service for bestshot frames.
func runWorker(ctx context.Context, q *memQueue, c *Global, s *Stats, hub *Hub, store *ShotStore) {
	if c.SaveDir != "" {
		if err := os.MkdirAll(c.SaveDir, 0o755); err != nil {
			log.Printf("save dir: %v", err)
		}
	}
	for {
		select {
		case <-ctx.Done():
			return
		case f := <-q.ch:
			if c.Rec != nil && f.Reason == "bestshot" {
				recRes, err := c.Rec.Recognize(ctx, f.JPEG, f.CamID, f.TrackID, f.Seq)
				if err == nil && (recRes.Status == "no face found" || recRes.Status == "no_face") {
					log.Printf("[%s] track=%d seq=%d -> no face detected by recognizer; purging from gallery",
						f.CamID, f.TrackID, f.Seq)
					if store != nil {
						store.RemoveShot(f.CamID, f.TrackID, f.Seq)
						store.BroadcastGallery(hub, true)
					}
					s.Processed.Add(1)
					continue
				}

				ev := RecogEvent{
					Type: "recog",
					Cam:  f.CamID,
					ID:   f.TrackID,
					Seq:  f.Seq,
					T:    time.Now().UnixMilli(),
				}
				if err != nil {
					log.Printf("[%s] recognition error (track=%d seq=%d): %v", f.CamID, f.TrackID, f.Seq, err)
					ev.Status = "error"
				} else {
					ev.Status = recRes.Status
					ev.Name = recRes.Name
					ev.Similarity = recRes.Similarity
					ev.Closest = recRes.Closest
					log.Printf("[%s] track=%d seq=%d -> status=%s name=%s sim=%.3f closest=%s",
						f.CamID, f.TrackID, f.Seq, ev.Status, ev.Name, ev.Similarity, ev.Closest)
				}
				if b, err := json.Marshal(ev); err == nil {
					hub.PublishRecog(b)
				}
			} else {
				if c.YoloMS > 0 {
					time.Sleep(time.Duration(c.YoloMS) * time.Millisecond)
				}
			}

			if c.SaveDir != "" {
				name := filepath.Join(c.SaveDir, fmt.Sprintf("%s_%06d_%s.jpg", f.CamID, f.Seq, f.Reason))
				_ = os.WriteFile(name, f.JPEG, 0o644)
			}
			s.Processed.Add(1)
		}
	}
}
