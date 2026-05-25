package transport

import (
	"context"
	"errors"
	"net/http"
	"time"
)

type RetryTransport struct {
	Base    http.RoundTripper
	Retries int
}

func (t RetryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.Base
	if base == nil {
		base = http.DefaultTransport
	}
	attempts := t.Retries
	if attempts <= 0 {
		attempts = 3
	}
	var last error
	backoff := 50 * time.Millisecond
	for attempt := 0; attempt < attempts; attempt++ {
		ctx, cancel := context.WithTimeout(req.Context(), 2*time.Second)
		cloned := req.Clone(ctx)
		resp, err := base.RoundTrip(cloned)
		cancel()
		if err == nil && resp.StatusCode < 500 {
			return resp, nil
		}
		if err != nil {
			last = err
		}
		time.Sleep(backoff)
		backoff *= 2
	}
	if last == nil {
		last = errors.New("retry transport exhausted")
	}
	return nil, last
}
