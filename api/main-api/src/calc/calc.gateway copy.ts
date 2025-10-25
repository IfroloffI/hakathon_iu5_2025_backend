import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Calc } from './schemas/calc.schema';
import { CalcService } from './calc.service';
import { v4 as uuidv4 } from 'uuid';

@WebSocketGateway({
  namespace: '/api/calc',
  cors: {
    origin: process.env.CORS_ORIGIN || '*',
    methods: ['GET', 'POST'],
    credentials: true,
  },
  transports: ['websocket', 'polling'],
})
export class CalcGateway {
  @WebSocketServer()
  server: Server;

  constructor(
    @InjectModel(Calc.name) private jobModel: Model<Calc>,
    private calcService: CalcService
  ) {}

  @SubscribeMessage('calculate')
  async handleCalculate(
    @MessageBody() payload: { observations: any[] },
    @ConnectedSocket() client: Socket
  ) {
    if (!payload.observations || payload.observations.length < 5) {
      client.emit('error', 'At least 5 observations required');
      return;
    }

    const job = new this.jobModel({
      userId: 'anonymous', // ← замените на req.user.userId
      status: 'queued',
      observations: payload.observations,
    });
    await job.save();

    const jobId = job._id.toString();
    client.emit('status', { jobId, status: 'queued' });

    // TODO: [Очередь] Отправлять задачу в Redis Streams / RabbitMQ вместо прямого вызова
    // Сейчас: this.processJob(...)
    // Будет: this.queueService.addJob({ jobId, observations })

    this.processJob(jobId, client.id).catch(console.error);
  }

  private async processJob(jobId: string, socketId: string) {
    // TODO: [Очередь] Этот метод должен быть в отдельном воркере (worker service),
    // который слушает очередь, а не вызывается напрямую из WebSocket.
    // Это позволит масштабировать обработку независимо от API.

    await this.jobModel.findByIdAndUpdate(jobId, { status: 'processing' });
    this.server.to(socketId).emit('status', { jobId, status: 'processing' });

    try {
      const job = await this.jobModel.findById(jobId);
      if (!job || job.status === 'deleted') return;

      const result = await this.calcService.calculateOrbit(job.observations);
      if (typeof result !== 'object' || result === null) {
        throw new Error('Invalid result from gRPC service');
      }

      await this.jobModel.findByIdAndUpdate(jobId, {
        status: 'completed',
        ...result,
      });

      this.server.to(socketId).emit('result', { jobId, ...result });
    } catch (error) {
      const errResult = { success: false, error: error.message };
      await this.jobModel.findByIdAndUpdate(jobId, {
        status: 'completed',
        ...errResult,
      });
      this.server.to(socketId).emit('result', { jobId, ...errResult });
    }
  }
}
