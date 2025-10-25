Calc Service gRPC Generate:

python -m venv venv

venv\Scripts\Activate.ps1

python -m grpc_tools.protoc -I ../contracts --python_out=. --grpc_python_out=. ../contracts/calc.proto

python .\server.py

Main Service gRPC Generate:

mkdir -p api/main-api/src/calc/assets/proto

cp api/contracts/calc.proto api/main-api/src/calc/assets/proto/

