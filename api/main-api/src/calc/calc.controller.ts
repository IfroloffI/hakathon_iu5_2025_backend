import {
    Controller,
    Post,
    Body,
    UseGuards,
    HttpException,
    HttpStatus,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { CalcService } from './calc.service';
import { CalcRequestDto } from './dto/calc.dto';

@Controller('api/calc')
// @UseGuards(AuthGuard('jwt'))
export class CalcController {
    constructor(private readonly calcService: CalcService) { }

    @Post()
    async calculate(@Body() body: CalcRequestDto) {
        if (!body.observations || body.observations.length < 5) {
            throw new Error('At least 5 observations required');
        }
        return this.calcService.calculateOrbit(body.observations);
    }
}
