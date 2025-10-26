## Calc-api setup

python -m venv venv

venv\Scripts\Activate.ps1

### Activate venv
**Windows:**
`venv\Scripts\Activate.ps1
`

**Mac:**
`source venv/bin/activate`

#### Gen proto
`python -m grpc_tools.protoc -I ../contracts --python_out=. --grpc_python_out=. ../contracts/calc.proto
`
#### Start server
`python .\server.py
`
## Main Service gRPC Generate:

mkdir -p api/main-api/src/calc/assets/proto

cp api/contracts/calc.proto api/main-api/src/calc/assets/proto/

## Деплой:

docker compose up -d mongodb redis calc-api
docker compose up -d --build main-api
