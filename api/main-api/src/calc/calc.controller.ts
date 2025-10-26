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
  async calculate(@Body() body: CalcRequestDto, @Req() req: RequestWithUser) {
    if (!body.observations || body.observations.length < 3) {
      throw new BadRequestException('At least 3 observations required');
    }

    // Сохраняем в БД
    const job = await this.calcService.calcModel.create({
      userId: req.user.userId,
      observations: body.observations,
      status: 'pending',
    });

    // Отправляем в Redis ВСЕ данные
    await this.redis.xadd('calculation_jobs', '*', {
      jobId: job._id.toString(),
      socketId: 'temp', // замени на реальный, если есть
      obs: JSON.stringify(body.observations),
    });

    return { jobId: job._id.toString() };
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
