# --- build stage ---
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
# static, stripped binary
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /tf-anthropic-proxy .

# --- runtime stage: distroless static (~2MB, CA certs, non-root) ---
FROM gcr.io/distroless/static:nonroot
COPY --from=build /tf-anthropic-proxy /tf-anthropic-proxy
USER nonroot:nonroot
ENV LISTEN_ADDR=:4000
EXPOSE 4000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/tf-anthropic-proxy", "-health"]
ENTRYPOINT ["/tf-anthropic-proxy"]
