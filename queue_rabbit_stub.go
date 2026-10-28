//go:build !rabbitmq

package main

import "errors"

func newRabbitQueue(url, name string) (Queue, error) {
	return nil, errors.New("built without RabbitMQ: run `go get github.com/rabbitmq/amqp091-go` and build with -tags rabbitmq")
}
