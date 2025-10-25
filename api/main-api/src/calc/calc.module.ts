import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { CalcGateway } from './calc.gateway';
import { CalcService } from './calc.service';
import { CalcController } from './calc.controller';
import { Calc, CalcSchema } from './schemas/calc.schema';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: Calc.name, schema: CalcSchema }]),
  ],
  controllers: [CalcController],
  providers: [CalcGateway, CalcService],
})
export class CalcModule {}
