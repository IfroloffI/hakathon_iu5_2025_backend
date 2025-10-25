import { Test, TestingModule } from '@nestjs/testing';
import { CalcService } from '../../src/calc/calc.service';
import * as grpc from '@grpc/grpc-js';

const mockGrpcClient = {
    CalculateOrbit: jest.fn(),
};

jest.mock('@grpc/grpc-js', () => {
    const actual = jest.requireActual('@grpc/grpc-js');
    return {
        ...actual,
        loadPackageDefinition: jest.fn().mockReturnValue({
            calc: {
                CometCalculator: jest.fn(() => mockGrpcClient),
            },
        }),
    };
});

describe('CalcService', () => {
    let service: CalcService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [CalcService],
        }).compile();

        service = module.get(CalcService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    it('should return orbit data', async () => {
        const mockResponse = {
            success: true,
            semi_major_axis_au: 3.2,
            eccentricity: 0.65,
            inclination_deg: 12.5,
            longitude_ascending_node_deg: 45.0,
            argument_perihelion_deg: 80.0,
            perihelion_passage_jd: 2460500.5,
            closest_approach_jd: 2460600.5,
            closest_distance_au: 0.8,
        };

        mockGrpcClient.CalculateOrbit.mockImplementation((_, cb) => cb(null, mockResponse));

        const result = await service.calculateOrbit([
            { ra_hours: 5.1, dec_degrees: 22.1, timestamp: 1729843200 },
            { ra_hours: 5.2, dec_degrees: 22.2, timestamp: 1729929600 },
            { ra_hours: 5.3, dec_degrees: 22.3, timestamp: 1730016000 },
            { ra_hours: 5.4, dec_degrees: 22.4, timestamp: 1730102400 },
            { ra_hours: 5.5, dec_degrees: 22.5, timestamp: 1730188800 },
        ]);

        expect(result).toEqual(mockResponse);
    });
});