import { Injectable } from '@nestjs/common';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';

const PROTO_PATH = __dirname + '/assets/proto/calc.proto';

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
});

const calcProto: any = grpc.loadPackageDefinition(packageDefinition).calc;

@Injectable()
export class CalcService {
    private client: any;

    constructor() {
        const host = process.env.CALC_API_HOST || 'calc-api:50051';
        this.client = new calcProto.CometCalculator(
            host,
            grpc.credentials.createInsecure(),
        );
    }

    async calculateOrbit(
        observations: { ra_hours: number; dec_degrees: number; timestamp: number }[]
    ) {
        return new Promise((resolve) => {
            this.client.CalculateOrbit({ observations }, (err, response) => {
                if (err) {
                    resolve({ success: false, error: err.message });
                } else {
                    resolve(response);
                }
            });
        });
    }
}
