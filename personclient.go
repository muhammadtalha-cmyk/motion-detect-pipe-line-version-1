package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// PersonClient talks to mediapipe_service.py (MediaPipe has no Go version).
type PersonClient struct {
	url string
	hc  *http.Client
}

// FaceBox is the face bounding box returned by mediapipe_service.py
type FaceBox struct {
	X     int     `json:"x"`
	Y     int     `json:"y"`
	W     int     `json:"w"`
	H     int     `json:"h"`
	Score float64 `json:"score"`
	LX    float64 `json:"lx"` // left eye normalized x (0 if missing)
	LY    float64 `json:"ly"`
	RX    float64 `json:"rx"` // right eye normalized x (0 if missing)
	RY    float64 `json:"ry"`
}

type PersonBox struct {
	X     int      `json:"x"`
	Y     int      `json:"y"`
	W     int      `json:"w"`
	H     int      `json:"h"`
	Score float64  `json:"score"`
	Face  *FaceBox `json:"face"` // nil when no face detected
}

func NewPersonClient(url string) *PersonClient {
	return &PersonClient{url: strings.TrimRight(url, "/"), hc: &http.Client{Timeout: 3 * time.Second}}
}

func (p *PersonClient) Health() error {
	resp, err := p.hc.Get(p.url + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}

// Detect sends one JPEG and returns the people MediaPipe found in it.
func (p *PersonClient) Detect(ctx context.Context, jpg []byte) ([]PersonBox, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url+"/detect", bytes.NewReader(jpg))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "image/jpeg")
	resp, err := p.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 200))
		return nil, fmt.Errorf("mediapipe service %d: %s", resp.StatusCode, b)
	}
	var out []PersonBox
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}
