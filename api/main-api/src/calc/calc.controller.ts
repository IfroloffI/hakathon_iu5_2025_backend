import {
  Controller,
  Post,
  Get,
  Body,
  UseGuards,
  Req,
  BadRequestException,
  Delete,
  Param,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { CalcService } from './calc.service';
import { CalcRequestDto } from './dto/calc.dto';
import { RequestWithUser } from '../auth/types';

@Controller('api/calc')
@UseGuards(AuthGuard('jwt'))
export class CalcController {
  constructor(private readonly calcService: CalcService) {}

  @Post()
  async calculate(@Body() body: CalcRequestDto) {
    if (!body.observations || body.observations.length < 5) {
      throw new BadRequestException('At least 5 observations required');
    }
    return this.calcService.calculateOrbitOld(body.observations);
  }

  @Get('list')
  async getCalcList(@Req() req: RequestWithUser) {
    return this.calcService.findAllByUserId(req.user.userId);
  }

  @Get(':id')
  async getCalcById(@Req() req: RequestWithUser, @Param('id') id: string) {
    if (!id) throw new BadRequestException('ID is required');
    return this.calcService.findByIdAndUserId(id, req.user.userId);
  }

  @Delete(':id')
  async deleteCalcById(@Req() req: RequestWithUser, @Param('id') id: string) {
    if (!id) throw new BadRequestException('ID is required');
    await this.calcService.softDeleteByIdAndUserId(id, req.user.userId);
    return { message: 'Successfully deleted' };
  }
}
