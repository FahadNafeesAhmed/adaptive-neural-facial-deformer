# CPU inference image: solve raw scans -> USD rig anywhere (workstation, CI, AWS Batch/ECS).
#   docker build -t anfd .
#   docker run --rm -v "$PWD/scans:/in" -v "$PWD/out:/out" anfd /in -o /out/performance.usda --method neural
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 ANFD_MODEL=/app/data/ict_model.npz
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data/ict_model.npz /app/data/ict_model.npz
ARG CKPT=weights/solver.pt
COPY ${CKPT} /app/model/solver.pt

ENTRYPOINT ["anfd-solve", "--checkpoint", "/app/model/solver.pt"]
