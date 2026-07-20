# --- build stage ---
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
# static binary so the runtime image can stay tiny
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /tf-anthropic-proxy .

# --- runtime stage ---
FROM alpine:3.20
RUN apk add --no-cache ca-certificates wget \
    && adduser -D -u 10001 app
COPY --from=build /tf-anthropic-proxy /usr/local/bin/tf-anthropic-proxy
USER app
ENV LISTEN_ADDR=:4000
EXPOSE 4000
ENTRYPOINT ["tf-anthropic-proxy"]
