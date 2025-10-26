import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { CalcGateway } from './calc.gateway';
import { CalcService } from './calc.service';
import { CalcController } from './calc.controller';
import { Calc, CalcSchema } from './schemas/calc.schema';
import { JwtModule } from '@nestjs/jwt';
import { AuthModule } from '../auth/auth.module';
import { CalcWorkerService } from './calc-worker.service';
import { NotificationService } from './notification.service';
import { ConfigModule, ConfigService } from '@nestjs/config';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: Calc.name, schema: CalcSchema }]),
    JwtModule.registerAsync({
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: (config: ConfigService) => ({
        secret: config.get<string>('JWT_SECRET', 'fallback-secret-for-dev'),
      }),
    }),
    JwtModule,
    AuthModule,
  ],
  controllers: [CalcController],
  providers: [CalcGateway, CalcService, CalcWorkerService, NotificationService],
})
export class CalcModule {}
