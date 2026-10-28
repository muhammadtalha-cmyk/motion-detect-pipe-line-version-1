package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type RecResult struct {
	ID         int     `json:"id"`
	Status     string  `json:"status"`
	Name       string  `json:"name"`
	Similarity float64 `json:"similarity"`
	Second     float64 `json:"second"`
	Closest    string  `json:"closest"`
	DetScore   float64 `json:"det_score"`
	FacePx     int     `json:"face_px"`
}

type RecClient struct {
	url string
	hc  *http.Client
}

func NewRecClient(rawURL string) *RecClient {
	return &RecClient{
		url: strings.TrimRight(rawURL, "/"),
		hc:  &http.Client{Timeout: 30 * time.Second},
	}
}

func (r *RecClient) Health() error {
	resp, err := r.hc.Get(r.url + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}

func (r *RecClient) Recognize(ctx context.Context, jpg []byte, cam string, track int, seq uint64) (*RecResult, error) {
	q := url.Values{}
	q.Set("cam", cam)
	q.Set("track", strconv.Itoa(track))
	q.Set("seq", strconv.FormatUint(seq, 10))

	reqURL := fmt.Sprintf("%s/recognize?%s", r.url, q.Encode())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(jpg))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "image/jpeg")

	resp, err := r.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		return nil, fmt.Errorf("recognition service HTTP %d: %s", resp.StatusCode, string(b))
	}

	var out RecResult
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}
