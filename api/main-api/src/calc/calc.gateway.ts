import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket, BroadcastOperator, DefaultEventsMap } from 'socket.io';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Calc } from './schemas/calc.schema';
import { CalcService } from './calc.service';
import { CalcRequestDto } from './dto/calc.dto';
import { validate } from 'class-validator';
import { plainToInstance } from 'class-transformer';

type EventEmitter = Server | Socket | BroadcastOperator<DefaultEventsMap, any>;

@WebSocketGateway({ namespace: '/api/calc' })
export class CalcGateway {
  @WebSocketServer()
  server: Server;

  constructor(
    @InjectModel(Calc.name) private calcModel: Model<Calc>,
    private calcService: CalcService
  ) {}

  private sendJsonEvent(emitter: EventEmitter, event: string, data: any): void {
    const payload = typeof data === 'object' && data !== null ? data : { data };
    emitter.emit(event, JSON.stringify(payload)); // TODO: после тестов обратно в бинарный payload без обёртки
  }

  @SubscribeMessage('calculate')
  async handleCalculate(
    @MessageBody() rawPayload: any,
    @ConnectedSocket() client: Socket
  ) {
    let payload: any;
    if (typeof rawPayload === 'string') {
      try {
        payload = JSON.parse(rawPayload);
      } catch {
        this.sendJsonEvent(client, 'error', { message: 'Invalid JSON format' });
        return;
      }
    } else if (rawPayload && typeof rawPayload === 'object') {
      payload = rawPayload;
    } else {
      this.sendJsonEvent(client, 'error', { message: 'Expected JSON object' });
      return;
    }

    const dto = plainToInstance(CalcRequestDto, payload);
    const errors = await validate(dto);
    if (errors.length > 0) {
      const messages = errors.flatMap((err) =>
        Object.values(err.constraints || {})
      );
      this.sendJsonEvent(client, 'error', { message: messages.join('; ') });
      return;
    }

    try {
      const job = new this.calcModel({
        userId: 'anonymous',
        status: 'queued',
        observations: dto.observations,
      });
      await job.save();

      const jobId = job._id.toString();
      this.sendJsonEvent(client, 'status', { jobId, status: 'queued' });

      // TODO: [Очередь] Отправлять задачу в Redis Streams / RabbitMQ вместо прямого вызова
      // Сейчас: this.processJob(...)
      // Будет: this.queueService.addJob({ jobId, observations })

      this.processJob(jobId, client.id).catch((err) => {
        console.error(`[CalcGateway] Background job failed for ${jobId}:`, err);
      });
    } catch (error) {
      console.error('[CalcGateway] Failed to create job:', error);
      this.sendJsonEvent(client, 'error', {
        message: 'Failed to create calculation job',
      });
    }
  }

  // TODO: [Очередь] Этот метод должен быть в отдельном воркере (worker service),
  // который слушает очередь, а не вызывается напрямую из WebSocket.
  // Это позволит масштабировать обработку независимо от API.
  private async processJob(jobId: string, socketId: string) {
    try {
      await this.calcModel.findByIdAndUpdate(jobId, { status: 'processing' });
      this.sendJsonEvent(this.server.to(socketId), 'status', {
        jobId,
        status: 'processing',
      });

      const job = await this.calcModel.findById(jobId);
      if (!job || job.status === 'deleted') return;

      const result = await this.calcService.calculateOrbit(job.observations);
      if (typeof result !== 'object' || result === null) {
        throw new Error('Invalid response from gRPC service');
      }

      await this.calcModel.findByIdAndUpdate(jobId, {
        status: 'completed',
        ...result,
      });
      this.sendJsonEvent(this.server.to(socketId), 'result', {
        jobId,
        ...result,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';
      const errorResult = { success: false, error: errorMessage };

      await this.calcModel.findByIdAndUpdate(jobId, {
        status: 'failed',
        ...errorResult,
      });
      this.sendJsonEvent(this.server.to(socketId), 'result', {
        jobId,
        ...errorResult,
      });
    }
  }
}
