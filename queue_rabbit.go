//go:build rabbitmq

package main

import (
	"context"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

type rabbitQueue struct {
	conn *amqp.Connection
	ch   *amqp.Channel
	name string
}

func newRabbitQueue(url, name string) (Queue, error) {
	conn, err := amqp.Dial(url)
	if err != nil {
		return nil, err
	}
	ch, err := conn.Channel()
	if err != nil {
		return nil, err
	}
	if _, err = ch.QueueDeclare(name, true, false, false, false, nil); err != nil {
		return nil, err
	}
	return &rabbitQueue{conn, ch, name}, nil
}

func (q *rabbitQueue) Publish(ctx context.Context, f Frame) error {
	return q.ch.PublishWithContext(ctx, "", q.name, false, false, amqp.Publishing{
		ContentType:  "image/jpeg",
		DeliveryMode: amqp.Persistent,
		Timestamp:    time.Now(),
		Headers:      amqp.Table{"cam_id": f.CamID, "seq": int64(f.Seq), "reason": f.Reason, "tracks": f.Tracks},
		Body:         f.JPEG,
	})
}
func (q *rabbitQueue) Depth() int { return -1 } // see RabbitMQ management UI
func (q *rabbitQueue) Close() error {
	q.ch.Close()
	return q.conn.Close()
}
