# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Lambda Layer builder for rawpy + numpy + Pillow + libraw
#
# Build:
#   docker build -t photo-editor-layer .
#   docker run --rm -v $(pwd)/layer-output:/output photo-editor-layer
#
# The container copies a ready-to-zip "python" directory into /output.
# Zip it and upload to Lambda as a layer:
#   cd layer-output && zip -r9 ../photo-layer.zip python/
#   aws lambda publish-layer-version \
#     --layer-name photo-editor-deps \
#     --zip-file fileb://../photo-layer.zip \
#     --compatible-runtimes python3.11 \
#     --compatible-architectures arm64
# ─────────────────────────────────────────────────────────────────────────────

FROM public.ecr.aws/lambda/python:3.11-arm64 AS builder

RUN yum install -y \
    gcc gcc-c++ make \
    cmake \
    zlib-devel \
    libjpeg-devel \
    jasper-devel \
    lcms2-devel \
    && yum clean all

# ── Build libraw from source (latest stable) ─────────────────────────────────
ARG LIBRAW_VERSION=0.21.2
WORKDIR /tmp
RUN curl -fsSL "https://www.libraw.org/data/LibRaw-${LIBRAW_VERSION}.tar.gz" \
    | tar xz && \
    cd LibRaw-${LIBRAW_VERSION} && \
    ./configure --prefix=/usr/local --disable-examples --disable-openmp && \
    make -j$(nproc) && make install && \
    ldconfig /usr/local/lib

# ── Install Python packages into the layer directory ────────────────────────
ENV LAYER_DIR=/opt/python
RUN mkdir -p ${LAYER_DIR}

RUN pip install --upgrade pip && \
    pip install \
      rawpy==0.21.0 \
      numpy==1.26.4 \
      Pillow==10.3.0 \
    --target ${LAYER_DIR} \
    --no-cache-dir \
    --platform linux_aarch64 \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: || \
    pip install \
      rawpy==0.21.0 \
      numpy==1.26.4 \
      Pillow==10.3.0 \
    --target ${LAYER_DIR} \
    --no-cache-dir

# Copy libraw shared libraries so rawpy can find them at runtime
RUN cp /usr/local/lib/libraw*.so* ${LAYER_DIR}/ 2>/dev/null || true

# Strip debug symbols to reduce layer size
RUN find ${LAYER_DIR} -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true
RUN find ${LAYER_DIR} -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
RUN find ${LAYER_DIR} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN find ${LAYER_DIR} -name "*.pyc" -delete 2>/dev/null || true

# ── Export stage ─────────────────────────────────────────────────────────────
FROM scratch AS export
COPY --from=builder /opt /opt

# ── Runner: copies layer to host-mounted /output ─────────────────────────────
FROM builder AS runner
ENTRYPOINT ["sh", "-c", "cp -r /opt/python /output/python && echo 'Layer ready in /output/python'"]