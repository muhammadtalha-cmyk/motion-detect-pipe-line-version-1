package main

import "context"

// Queue is the hand-off between the motion gate and YOLO.
type Queue interface {
	Publish(ctx context.Context, f Frame) error
	Depth() int // -1 if unknown
	Close() error
}

// memQueue: bounded in-process queue. If YOLO falls far behind and it fills up,
// the OLDEST frame is discarded (counted in Stats.Overflow) so the gate never stalls.
type memQueue struct {
	ch    chan Frame
	stats *Stats
}

func newMemQueue(n int, s *Stats) *memQueue { return &memQueue{ch: make(chan Frame, n), stats: s} }

func (q *memQueue) Publish(_ context.Context, f Frame) error {
	for {
		select {
		case q.ch <- f:
			return nil
		default:
		}
		select {
		case <-q.ch:
			q.stats.Overflow.Add(1)
		default:
		}
	}
}
func (q *memQueue) Depth() int  { return len(q.ch) }
func (q *memQueue) Close() error { return nil }
