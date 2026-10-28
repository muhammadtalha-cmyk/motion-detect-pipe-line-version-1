package main

import (
	"math"
	"sort"
)

// Detection is a body box plus an optional face box passed to the tracker each frame.
type Detection struct {
	Box  Box
	Face *Box
}

type TrackBox struct {
	ID   int  `json:"id"`
	X    int  `json:"x"`
	Y    int  `json:"y"`
	W    int  `json:"w"`
	H    int  `json:"h"`
	Face *Box `json:"face,omitempty"`
}

type track struct {
	id, hits, missed int
	box              Box
	face             *Box
}

// Tracker gives moving boxes a consistent ID by matching each new box to the
// existing track it overlaps most (IoU). No YOLO needed: it tracks MOTION blobs.
type Tracker struct {
	iou       float64
	maxMissed int // frames a track may go unmatched before it is deleted
	minHits   int // matches needed before a track gets a real ID (filters flicker)
	nextID    int
	tracks    []*track
}

func NewTracker(iou float64, maxMissed, minHits int) *Tracker {
	return &Tracker{iou: iou, maxMissed: maxMissed, minHits: minHits, nextID: 1}
}

func calcIoU(a, b Box) float64 {
	x1, y1 := max(a.X, b.X), max(a.Y, b.Y)
	x2, y2 := min(a.X+a.W, b.X+b.W), min(a.Y+a.H, b.Y+b.H)
	iw, ih := x2-x1, y2-y1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	union := float64(a.W*a.H+b.W*b.H) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// matchMetric matches detections to tracks using IoU, falling back to proximity for fast motion.
func matchMetric(a, b Box, minIoU float64) float64 {
	iou := calcIoU(a, b)
	if iou >= minIoU {
		return iou
	}
	// Fallback for fast motion: normalized center distance
	caX, caY := float64(a.X)+float64(a.W)/2, float64(a.Y)+float64(a.H)/2
	cbX, cbY := float64(b.X)+float64(b.W)/2, float64(b.Y)+float64(b.H)/2
	dx := caX - cbX
	dy := caY - cbY
	dist := math.Sqrt(dx*dx + dy*dy)
	maxDim := math.Max(float64(a.W+b.W), float64(a.H+b.H))
	if maxDim > 0 && dist < maxDim*1.2 {
		return 0.20 * (1.0 - dist/(maxDim*1.2))
	}
	return iou
}

// Update takes this frame's detections (body+face pairs) and returns confirmed tracks.
func (t *Tracker) Update(dets []Detection) []TrackBox {
	type pair struct {
		ti, bi int
		v      float64
	}
	var pairs []pair
	for ti, tr := range t.tracks {
		for bi, d := range dets {
			if v := matchMetric(tr.box, d.Box, t.iou); v >= 0.05 {
				pairs = append(pairs, pair{ti, bi, v})
			}
		}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].v > pairs[j].v })

	usedT := make([]bool, len(t.tracks))
	usedB := make([]bool, len(dets))
	for _, p := range pairs {
		if usedT[p.ti] || usedB[p.bi] {
			continue
		}
		usedT[p.ti], usedB[p.bi] = true, true
		tr := t.tracks[p.ti]
		tr.box = dets[p.bi].Box
		tr.face = dets[p.bi].Face
		tr.hits++
		tr.missed = 0
	}

	alive := make([]*track, 0, len(t.tracks)+len(dets))
	for i, tr := range t.tracks {
		if !usedT[i] {
			tr.missed++
		}
		if tr.missed <= t.maxMissed {
			alive = append(alive, tr)
		}
	}
	for bi, d := range dets {
		if !usedB[bi] {
			alive = append(alive, &track{box: d.Box, face: d.Face, hits: 1})
		}
	}
	t.tracks = alive

	var out []TrackBox
	for _, tr := range t.tracks {
		if tr.missed != 0 || tr.hits < t.minHits {
			continue
		}
		if tr.id == 0 {
			tr.id = t.nextID
			t.nextID++
		}
		out = append(out, TrackBox{tr.id, tr.box.X, tr.box.Y, tr.box.W, tr.box.H, tr.face})
	}
	return out
}
