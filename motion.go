package main

import (
	"image"
	"image/color"
)

type Box struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

type Result struct {
	Motion bool
	Score  float64 // fraction of changed pixels
	Boxes  []Box   // in full-frame pixel coordinates
}

// Detector compares the current frame (k+1) with the frame two steps back (k-1).
type Detector struct {
	tau, step, block int
	theta, blockFrac float64
	dw, dh           int
	older, newer, cur []uint8
	seen             int
}

func NewDetector(tau int, theta float64, step, block int, blockFrac float64) *Detector {
	return &Detector{tau: tau, theta: theta, step: step, block: block, blockFrac: blockFrac}
}

func (d *Detector) Process(img image.Image) Result {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	dw, dh := W/d.step, H/d.step
	if dw != d.dw || dh != d.dh {
		d.dw, d.dh = dw, dh
		n := dw * dh
		d.older, d.newer, d.cur = make([]uint8, n), make([]uint8, n), make([]uint8, n)
		d.seen = 0
	}
	d.downsample(img, b)

	var res Result
	if d.seen >= 2 {
		bw, bh := (dw+d.block-1)/d.block, (dh+d.block-1)/d.block
		counts := make([]int, bw*bh)
		changed := 0
		for y := 0; y < dh; y++ {
			for x := 0; x < dw; x++ {
				i := y*dw + x
				diff := int(d.cur[i]) - int(d.older[i])
				if diff < 0 {
					diff = -diff
				}
				if diff > d.tau {
					changed++
					counts[(y/d.block)*bw+x/d.block]++
				}
			}
		}
		res.Score = float64(changed) / float64(dw*dh)
		res.Motion = res.Score > d.theta
		if res.Motion {
			res.Boxes = d.boxes(counts, bw, bh, W, H)
		}
	}
	// rotate: older <- newer <- cur
	d.older, d.newer, d.cur = d.newer, d.cur, d.older
	d.seen++
	return res
}

// downsample copies every step-th luma value into d.cur (no colour work needed).
func (d *Detector) downsample(img image.Image, b image.Rectangle) {
	switch v := img.(type) {
	case *image.YCbCr:
		for y := 0; y < d.dh; y++ {
			row := v.Y[(y*d.step)*v.YStride:]
			for x := 0; x < d.dw; x++ {
				d.cur[y*d.dw+x] = row[x*d.step]
			}
		}
	case *image.Gray:
		for y := 0; y < d.dh; y++ {
			row := v.Pix[(y*d.step)*v.Stride:]
			for x := 0; x < d.dw; x++ {
				d.cur[y*d.dw+x] = row[x*d.step]
			}
		}
	default:
		for y := 0; y < d.dh; y++ {
			for x := 0; x < d.dw; x++ {
				g := color.GrayModel.Convert(img.At(b.Min.X+x*d.step, b.Min.Y+y*d.step)).(color.Gray)
				d.cur[y*d.dw+x] = g.Y
			}
		}
	}
}

// boxes groups neighbouring moving blocks into bounding boxes (flood fill, 8-connected).
func (d *Detector) boxes(counts []int, bw, bh, W, H int) []Box {
	need := int(d.blockFrac * float64(d.block*d.block))
	if need < 1 {
		need = 1
	}
	active := make([]bool, len(counts))
	for i, c := range counts {
		active[i] = c >= need
	}
	unit := d.block * d.step
	var out []Box
	for s := range active {
		if !active[s] {
			continue
		}
		minX, minY, maxX, maxY := bw, bh, 0, 0
		stack := []int{s}
		active[s] = false
		for len(stack) > 0 {
			i := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			x, y := i%bw, i/bw
			if x < minX { minX = x }
			if x > maxX { maxX = x }
			if y < minY { minY = y }
			if y > maxY { maxY = y }
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := x+dx, y+dy
					if nx < 0 || ny < 0 || nx >= bw || ny >= bh {
						continue
					}
					j := ny*bw + nx
					if active[j] {
						active[j] = false
						stack = append(stack, j)
					}
				}
			}
		}
		bx, by := minX*unit, minY*unit
		w, h := (maxX-minX+1)*unit, (maxY-minY+1)*unit
		if bx+w > W { w = W - bx }
		if by+h > H { h = H - by }
		out = append(out, Box{bx, by, w, h})
	}
	return out
}
