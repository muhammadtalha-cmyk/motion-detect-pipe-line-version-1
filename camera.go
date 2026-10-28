package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

type CamStats struct{ In, Passed, Dropped atomic.Uint64 }

// uiEvery: when a camera is idle, push only every Nth frame to the browser (saves bandwidth).
const uiEvery = 3

// minFaceWidthPx: low floor for face width in pixels (narrower is skipped)
const minFaceWidthPx = 24

// trackCandidate holds the best frame seen so far for a single tracked person.
type trackCandidate struct {
	score        float64
	qualityScore float64
	jpeg         []byte
	seq          uint64
	face         *FaceBox
	faceBox      *Box
	bodyBox      Box
	lastSent     time.Time
	faceHits     int  // number of verified face detections for this track
	sent         bool // true if a best shot has already been sent for this track
	missed       int  // consecutive frames where this track was not seen in lastTracks

	maxFaceWidth int     // max face width seen across all frames
	framesScored int     // count of frames scored
	bestScore    float64 // highest final score seen
	hasCandidate bool    // true if at least one candidate frame was stored
	skipReason   string  // reason if no candidate was saved
}

// ShotStore accumulates sent best-shot gallery entries for the web UI.
type ShotStore struct {
	mu        sync.Mutex
	sentLog   []*SentBestShot // ring buffer of last 50 sent shots
	sentLogMu sync.Mutex
}

func (s *ShotStore) RemoveShot(camID string, trackID int, seq uint64) {
	s.sentLogMu.Lock()
	defer s.sentLogMu.Unlock()
	filtered := make([]*SentBestShot, 0, len(s.sentLog))
	for _, shot := range s.sentLog {
		if shot.CamID == camID && shot.TrackID == trackID && shot.Seq == seq {
			continue
		}
		filtered = append(filtered, shot)
	}
	s.sentLog = filtered
}

func (s *ShotStore) BroadcastGallery(hub *Hub, rec bool) {
	if !hub.HasSubs() {
		return
	}
	s.sentLogMu.Lock()
	log2 := make([]*SentBestShot, len(s.sentLog))
	copy(log2, s.sentLog)
	s.sentLogMu.Unlock()
	b, _ := json.Marshal(map[string]interface{}{
		"type":      "bestshots",
		"bestshots": log2,
		"rec":       rec,
	})
	hub.Broadcast(b)
}

// SentBestShot is what we show in the web UI gallery.
type SentBestShot struct {
	CamID   string  `json:"cam"`
	TrackID int     `json:"id"`
	Seq     uint64  `json:"seq"`
	Score   float64 `json:"score"`
	JPEG    []byte  `json:"jpg"` // base64 in JSON = cropped upper body
	SentAt  int64   `json:"ts"`  // unix ms
}

func newShotStore() *ShotStore {
	return &ShotStore{}
}

// scoreFrame computes the best-shot score for a frame (0–1).
// Returns finalScore, qualityScore (before sizeFactor), and individual components for debug logging.
func scoreFrame(faceBox *Box, bodyW, bodyH int, faceConf float64,
	faceCrop image.Image, face *FaceBox) (finalScore, qualityScore, frontalMult, lapVar, brightScore, sizeFactor float64) {

	if faceBox == nil || bodyW <= 0 || bodyH <= 0 || faceConf < 0.40 {
		return 0, 0, 0, 0, 0, 0
	}

	// --- 1. Face Confidence Gate ---
	confGate := math.Pow(faceConf, 1.8)

	// --- 2. Frontality Multiplier (Zero-Trust) ---
	frontalMult = 0.05
	if face != nil && face.LX != 0 && face.RX != 0 {
		eyeDist := math.Abs(face.LX - face.RX)
		if eyeDist < 0.03 {
			return 0, 0, 0, 0, 0, 0 // eyes merged into single point: not a frontal face
		}
		// Eye midpoint relative to crop center (frontal face has eyes centered ~0.5)
		eyeMidX := (face.LX + face.RX) / 2.0
		frontalMult = math.Max(0.15, 1.0-math.Abs(eyeMidX-0.5)*3.0)
	}

	// Face gate: both confidence and frontality must agree
	faceGate := confGate * frontalMult

	// --- 3. Sharpness (Laplacian variance on face crop) ---
	sharpScore := 0.5
	if faceCrop != nil {
		lapVar = laplacianVariance(faceCrop)
		sharpScore = math.Min(lapVar/400.0, 1.0)
	}

	// --- 4. Brightness (mean luma) ---
	brightScore = 0.5
	if faceCrop != nil {
		mean := meanBrightness(faceCrop)
		if mean < 60 {
			brightScore = mean / 60.0
		} else if mean > 200 {
			brightScore = math.Max(0, 1.0-(mean-200)/55.0)
		} else {
			brightScore = 1.0
		}
	}

	// Visual quality factors (weighted clarity and lighting)
	// Normalized weights from original (0.35 sharp + 0.25 bright = 0.60 total)
	visualQuality := 0.58*sharpScore + 0.42*brightScore

	// Quality score (before size factor) to check against minshot threshold
	qualityScore = faceGate * visualQuality

	// --- 5. Rescaled Absolute Face Size Factor (Multiplicative clamp) ---
	// clamp(faceWidthPx / 50.0, 0.3, 1.0): 45px gets ~0.9, 25px gets ~0.5
	sizeFactor = math.Max(0.3, math.Min(float64(faceBox.W)/50.0, 1.0))

	// Multiplicative final score: qualityScore * sizeFactor
	finalScore = qualityScore * sizeFactor
	return finalScore, qualityScore, frontalMult, lapVar, brightScore, sizeFactor
}

// laplacianVariance computes the variance of a Laplacian filter on a grayscale image.
// Higher = sharper.
func laplacianVariance(img image.Image) float64 {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	if W < 3 || H < 3 {
		return 0
	}
	gray := make([]float64, W*H)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := color.GrayModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.Gray)
			gray[y*W+x] = float64(c.Y)
		}
	}
	var sum, sum2 float64
	n := 0
	for y := 1; y < H-1; y++ {
		for x := 1; x < W-1; x++ {
			lap := -gray[(y-1)*W+x] - gray[(y+1)*W+x] - gray[y*W+x-1] - gray[y*W+x+1] + 4*gray[y*W+x]
			sum += lap
			sum2 += lap * lap
			n++
		}
	}
	if n == 0 {
		return 0
	}
	mean := sum / float64(n)
	return sum2/float64(n) - mean*mean
}

// meanBrightness returns the mean luma of an image (0-255).
func meanBrightness(img image.Image) float64 {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	var sum float64
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := color.GrayModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.Gray)
			sum += float64(c.Y)
		}
	}
	if W*H == 0 {
		return 128
	}
	return sum / float64(W*H)
}

// cropUpperBody crops the upper portion of the person box from the image.
// It returns a JPEG-encoded image of the upper body (top half + face).
func cropUpperBody(img image.Image, body Box, face *Box) ([]byte, error) {
	bounds := img.Bounds()
	W, H := bounds.Dx(), bounds.Dy()

	// Take the top half of the body box, but always include the face if present
	cropH := body.H / 2
	if face != nil {
		// extend to include full face with padding
		faceBottom := face.Y + face.H + 10
		if faceBottom-body.Y > cropH {
			cropH = faceBottom - body.Y
		}
	}
	// add 10% padding
	pad := int(float64(body.H) * 0.10)
	cropX := max(0, body.X-pad)
	cropY := max(0, body.Y-pad)
	cropW := min(W-cropX, body.W+2*pad)
	cropH2 := min(H-cropY, cropH+pad)

	sub := image.NewRGBA(image.Rect(0, 0, cropW, cropH2))
	draw.Draw(sub, sub.Bounds(), img, image.Point{bounds.Min.X + cropX, bounds.Min.Y + cropY}, draw.Src)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, sub, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// sendBestShot crops and publishes the best shot for a track.
func sendBestShot(ctx context.Context, q Queue, hub *Hub, store *ShotStore,
	cs *CamStats, camID string, trackID int, tc *trackCandidate,
	img image.Image, stats *Stats, g *Global) {

	cropJPEG, err := cropUpperBody(img, tc.bodyBox, tc.faceBox)
	if err != nil {
		return
	}
	f := Frame{
		Seq:     tc.seq,
		CamID:   camID,
		TrackID: trackID,
		JPEG:    cropJPEG,
		Reason:  "bestshot",
		Score:   tc.score,
	}
	if err := q.Publish(ctx, f); err != nil {
		log.Printf("[%s] bestshot publish failed: %v", camID, err)
		return
	}
	cs.Passed.Add(1)

	// Add to store for web UI gallery
	shot := &SentBestShot{
		CamID:   camID,
		TrackID: trackID,
		Seq:     tc.seq,
		Score:   tc.score,
		JPEG:    cropJPEG,
		SentAt:  time.Now().UnixMilli(),
	}
	store.sentLogMu.Lock()
	store.sentLog = append(store.sentLog, shot)
	if len(store.sentLog) > 50 {
		store.sentLog = store.sentLog[len(store.sentLog)-50:]
	}
	store.sentLogMu.Unlock()

	// Broadcast gallery update
	store.BroadcastGallery(hub, g != nil && g.Rec != nil)
}

// runCamera = one camera's whole pipeline:
//
//	read -> decode -> motion gate -> [MediaPipe person detection] -> tracker -> queue + web UI
//
// Without -mp it tracks motion blobs. With -mp, MediaPipe runs ONLY on frames the gate lets
// through, the tracker gives IDs to the PEOPLE it finds, and only frames with people are queued.
func runCamera(ctx context.Context, g *Global, c CamConfig, q Queue, hub *Hub, stats *Stats, store *ShotStore) {
	frames := make(chan Frame, 64)
	go func() { runSource(ctx, g, &c, frames); close(frames) }()

	det := NewDetector(c.Tau, c.Theta, c.Step, c.Block, c.BlockFrac)
	every := uint64(max(g.MPEvery, 1))
	minHits, maxMissed := 2, c.FPS
	if g.MP != nil {
		minHits, maxMissed = 1, max(c.FPS/int(every), 2)
	}
	trk := NewTracker(c.IOU, maxMissed, minHits)
	holdFrames := uint64(c.Hold * float64(c.FPS))

	var cs CamStats
	var lastMotion, lastPerson, eligible uint64
	var lastTracks []TrackBox
	var lastErr time.Time

	// candidates accumulates best frames per track ID (only used in face/best-shot mode)
	candidates := make(map[int]*trackCandidate)
	bestEvery := time.Duration(g.BestEvery) * time.Second
	if bestEvery <= 0 {
		bestEvery = 10 * time.Second
	}
	debugBestShot := os.Getenv("DEBUG_BESTSHOT") == "1" || os.Getenv("DEBUG_BESTSHOT") == "true"

	for f := range frames {
		cs.In.Add(1)
		img, err := jpeg.Decode(bytes.NewReader(f.JPEG))
		if err != nil {
			continue
		}
		res := det.Process(img)
		if res.Motion {
			lastMotion = f.Seq
		}
		switch {
		case res.Motion:
			f.Reason = "motion"
		case lastMotion > 0 && f.Seq-lastMotion <= holdFrames:
			f.Reason = "hold"
		}
		f.Score = res.Score
		state := f.Reason
		var tracks []TrackBox
		uiBoxes := res.Boxes

		if g.MP == nil {
			// No MediaPipe: convert motion boxes to detections
			dets := make([]Detection, len(res.Boxes))
			for i, b := range res.Boxes {
				dets[i] = Detection{Box: b}
			}
			tracks = trk.Update(dets)
			if state == "" {
				state = "idle"
			}
		} else {
			uiBoxes = nil // person mode: show only person boxes
			// look for people while the gate is open, and as long as a person keeps being found
			watching := f.Reason != "" || (g.KeepStill && lastPerson > 0 && f.Seq-lastPerson <= holdFrames)
			if watching {
				eligible++
				if eligible%every == 1%every {
					persons, err := g.MP.Detect(ctx, f.JPEG)
					if err != nil {
						if time.Since(lastErr) > 5*time.Second {
							log.Printf("[%s] MediaPipe service error: %v", c.ID, err)
							lastErr = time.Now()
						}
					} else {
						dets := make([]Detection, 0, len(persons))
						for _, p := range persons {
							pb := Box{p.X, p.Y, p.W, p.H}
							// face-only mode: skip people with no face
							if g.FaceOnly && p.Face == nil {
								continue
							}
							// only count people who are actually MOVING: their box must overlap motion
							// (or, while the gate is cooling down, an already-confirmed moving person)
							if !g.KeepStill && !overlapsAny(pb, res.Boxes) && (res.Motion || !overlapsTracks(pb, lastTracks)) {
								continue
							}
							var faceBox *Box
							if p.Face != nil {
								fb := Box{p.Face.X, p.Face.Y, p.Face.W, p.Face.H}
								faceBox = &fb
							}
							dets = append(dets, Detection{Box: pb, Face: faceBox})
						}
						lastTracks = trk.Update(dets)

						// In face mode: accumulate best shots per track
						if g.FaceOnly {
							for _, tr := range lastTracks {
								if tr.Face == nil {
									continue
								}
								// Find the person data for this track to get FaceBox with landmarks
								var matchedPerson *PersonBox
								for i := range persons {
									pb := Box{persons[i].X, persons[i].Y, persons[i].W, persons[i].H}
									if calcIoU(pb, Box{tr.X, tr.Y, tr.W, tr.H}) > 0.3 {
										matchedPerson = &persons[i]
										break
									}
								}
								if matchedPerson == nil || matchedPerson.Face == nil {
									continue
								}

								// Compute face crop for sharpness/brightness
								var faceCropImg image.Image
								fb := tr.Face
								if fb != nil && fb.W > 0 && fb.H > 0 {
									imgBounds := img.Bounds()
									fx := max(imgBounds.Min.X, imgBounds.Min.X+fb.X)
									fy := max(imgBounds.Min.Y, imgBounds.Min.Y+fb.Y)
									fw := min(fb.W, imgBounds.Dx()-fb.X)
									fh := min(fb.H, imgBounds.Dy()-fb.Y)
									if fw > 0 && fh > 0 {
										cropImg := image.NewRGBA(image.Rect(0, 0, fw, fh))
										draw.Draw(cropImg, cropImg.Bounds(), img,
											image.Point{fx, fy}, draw.Src)
										faceCropImg = cropImg
									}
								}

								// Always track candidate entry to accumulate statistics
								tc, exists := candidates[tr.ID]
								if !exists {
									tc = &trackCandidate{
										lastSent: time.Now(),
									}
									candidates[tr.ID] = tc
								}
								tc.missed = 0
								tc.framesScored++
								if tr.Face.W > tc.maxFaceWidth {
									tc.maxFaceWidth = tr.Face.W
								}

								finalScore, qualityScore, frontalMult, lapVar, brightScore, _ := scoreFrame(tr.Face, tr.W, tr.H,
									matchedPerson.Face.Score, faceCropImg, matchedPerson.Face)

								// 1. Low floor gate: face width under minFaceWidthPx (24px)
								if tr.Face.W < minFaceWidthPx {
									if debugBestShot {
										log.Printf("[bestshot debug] track=%d seq=%d w=%dpx (<%dpx floor, skipped) conf=%.2f front=%.2f sharp=%.1f bright=%.2f quality=%.3f score=%.3f",
											tr.ID, f.Seq, tr.Face.W, minFaceWidthPx, matchedPerson.Face.Score, frontalMult, lapVar, brightScore, qualityScore, finalScore)
									}
									if !tc.hasCandidate && tc.skipReason == "" {
										tc.skipReason = fmt.Sprintf("floor (max width %dpx < %dpx)", tr.Face.W, minFaceWidthPx)
									}
									continue
								}

								// 2. Check quality threshold (minshot) on qualityScore BEFORE sizeFactor
								if qualityScore < g.MinShot {
									if debugBestShot {
										log.Printf("[bestshot debug] track=%d seq=%d w=%dpx (quality < minshot, skipped) conf=%.2f front=%.2f sharp=%.1f bright=%.2f quality=%.3f score=%.3f",
											tr.ID, f.Seq, tr.Face.W, matchedPerson.Face.Score, frontalMult, lapVar, brightScore, qualityScore, finalScore)
									}
									if !tc.hasCandidate && tc.skipReason == "" {
										tc.skipReason = fmt.Sprintf("quality below minshot (best quality %.3f < %.2f)", qualityScore, g.MinShot)
									}
									continue
								}

								// 3. This frame qualifies as candidate!
								tc.faceHits++
								replaced := false
								if !tc.hasCandidate || finalScore > tc.score {
									tc.score = finalScore
									tc.qualityScore = qualityScore
									tc.jpeg = f.JPEG
									tc.seq = f.Seq
									tc.face = matchedPerson.Face
									tc.faceBox = tr.Face
									tc.bodyBox = Box{tr.X, tr.Y, tr.W, tr.H}
									tc.hasCandidate = true
									if finalScore > tc.bestScore {
										tc.bestScore = finalScore
									}
									replaced = true
								}

								if debugBestShot {
									log.Printf("[bestshot debug] track=%d seq=%d w=%dpx conf=%.2f front=%.2f sharp=%.1f bright=%.2f quality=%.3f score=%.3f replaced=%v",
										tr.ID, f.Seq, tr.Face.W, matchedPerson.Face.Score, frontalMult, lapVar, brightScore, qualityScore, finalScore, replaced)
								}

								// Periodic send only for someone standing in view longer than bestEvery seconds
								if !tc.sent && tc.hasCandidate && time.Since(tc.lastSent) >= bestEvery {
									sendBestShot(ctx, q, hub, store, &cs, c.ID, tr.ID, tc, img, stats, g)
									tc.lastSent = time.Now()
									tc.sent = true
									log.Printf("[track finalize] track=%d max_w=%dpx best_score=%.3f frames_scored=%d -> SENT (seq=%d, periodic)",
										tr.ID, tc.maxFaceWidth, tc.bestScore, tc.framesScored, tc.seq)
									tc.score = 0
									tc.hasCandidate = false
								}
							}

							// Detect tracks that have truly ended using a grace period
							activeIDs := make(map[int]bool)
							for _, tr := range lastTracks {
								activeIDs[tr.ID] = true
							}
							graceFrames := max(c.FPS/2, 8)
							for tid, tc := range candidates {
								if !activeIDs[tid] {
									tc.missed++
									if tc.missed >= graceFrames {
										// Finalize track: send if it has any candidate
										if !tc.sent {
											if tc.hasCandidate && len(tc.jpeg) > 0 {
												lastImg, err2 := jpeg.Decode(bytes.NewReader(tc.jpeg))
												if err2 == nil {
													sendBestShot(ctx, q, hub, store, &cs, c.ID, tid, tc, lastImg, stats, g)
													tc.sent = true
													log.Printf("[track finalize] track=%d max_w=%dpx best_score=%.3f frames_scored=%d -> SENT (seq=%d)",
														tid, tc.maxFaceWidth, tc.bestScore, tc.framesScored, tc.seq)
												}
											} else {
												reason := tc.skipReason
												if reason == "" {
													if tc.maxFaceWidth == 0 {
														reason = "no candidate (no face detected)"
													} else if tc.maxFaceWidth < minFaceWidthPx {
														reason = fmt.Sprintf("floor (max width %dpx < %dpx)", tc.maxFaceWidth, minFaceWidthPx)
													} else {
														reason = "quality below minshot"
													}
												}
												log.Printf("[track finalize] track=%d max_w=%dpx best_score=%.3f frames_scored=%d -> NOT SENT (%s)",
													tid, tc.maxFaceWidth, tc.bestScore, tc.framesScored, reason)
											}
										}
										delete(candidates, tid)
									}
								}
							}
						}
					}
				}
				tracks = lastTracks
			} else {
				trk.Update(nil)
				lastTracks = nil
			}
			if len(tracks) > 0 {
				lastPerson = f.Seq
				f.Reason, state = "person", "person"
			} else {
				f.Reason = "" // motion but no person: do not send
				if state == "" {
					state = "idle"
				}
			}
		}
		if len(tracks) > 0 {
			tj, _ := json.Marshal(tracks)
			f.Tracks = string(tj)
		}

		// In face/best-shot mode, we don't send individual frames to queue (sendBestShot does it).
		// In non-face mode, send as before.
		sent := false
		if !g.FaceOnly {
			if f.Reason != "" {
				if err := q.Publish(ctx, f); err != nil {
					log.Printf("[%s] publish failed: %v", c.ID, err)
				} else {
					sent = true
					cs.Passed.Add(1)
				}
			}
		}
		if !sent {
			cs.Dropped.Add(1)
		}

		if hub.HasSubs() && (sent || f.Seq%uiEvery == 0) {
			b, _ := json.Marshal(Event{
				Cam: c.ID, Seq: f.Seq, JPEG: f.JPEG, Boxes: uiBoxes, Tracks: tracks, Motion: res.Motion,
				Sent: sent, State: state, Reason: f.Reason, Score: res.Score, Theta: c.Theta,
				CS:    CamSnap{In: cs.In.Load(), Passed: cs.Passed.Load(), Dropped: cs.Dropped.Load()},
				Stats: Snapshot{Processed: stats.Processed.Load(), Overflow: stats.Overflow.Load(), Depth: q.Depth()},
				Rec:   g.Rec != nil,
			})
			hub.Broadcast(b)
		}
	}
}

// minOverlap: fraction of the person's box that must be covered by motion to count as moving.
const minOverlap = 0.05

func overlapFrac(p, m Box) float64 {
	x1, y1 := max(p.X, m.X), max(p.Y, m.Y)
	x2, y2 := min(p.X+p.W, m.X+m.W), min(p.Y+p.H, m.Y+m.H)
	if x2 <= x1 || y2 <= y1 || p.W*p.H <= 0 {
		return 0
	}
	return float64((x2-x1)*(y2-y1)) / float64(p.W*p.H)
}

func overlapsAny(p Box, motion []Box) bool {
	for _, m := range motion {
		if overlapFrac(p, m) >= minOverlap {
			return true
		}
	}
	return false
}

func overlapsTracks(p Box, ts []TrackBox) bool {
	for _, t := range ts {
		if overlapFrac(p, Box{t.X, t.Y, t.W, t.H}) >= 0.3 {
			return true
		}
	}
	return false
}
