FROM rust:slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --release --bin terrarium_web -p oneura-cli --features web

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgcc-s1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/terrarium_web /usr/local/bin/terrarium_web

ENV PORT=8420
ENV SEED=42
ENV FPS=10
ENV REQUIRE_AUTH=false

EXPOSE 8420

CMD ["terrarium_web"]
