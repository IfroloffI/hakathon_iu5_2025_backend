import { Test, TestingModule } from '@nestjs/testing';
import { CalcController } from '../../src/calc/calc.controller';
import { CalcService } from '../../src/calc/calc.service';
import { BadRequestException } from '@nestjs/common';

describe('CalcController', () => {
    let controller: CalcController;
    let service: CalcService;

    const mockService = {
        calculateOrbit: jest.fn(),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [CalcController],
            providers: [{ provide: CalcService, useValue: mockService }],
        }).compile();

        controller = module.get(CalcController);
        service = module.get(CalcService);
    });

    it('should require at least 5 observations', async () => {
        await expect(
            controller.calculate({ observations: [{ ra_hours: 1, dec_degrees: 1, timestamp: 1 }] })
        ).rejects.toThrow(BadRequestException);
    });

    it('should call service with valid input', async () => {
        const dto = {
            observations: Array(5).fill({ ra_hours: 1, dec_degrees: 1, timestamp: 1 }),
        };
        mockService.calculateOrbit.mockResolvedValue({ success: true });

        await controller.calculate(dto);

        expect(service.calculateOrbit).toHaveBeenCalledWith(dto.observations);
    });
});