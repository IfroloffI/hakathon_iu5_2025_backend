import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
  OnGatewayConnection,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Calc } from './schemas/calc.schema';
import { CalcRequestDto } from './dto/calc.dto';
import { validate } from 'class-validator';
import { JwtService } from '@nestjs/jwt';
import { plainToInstance } from 'class-transformer';
import { WhitelistService } from '../auth/whitelist.service';
import { RedisService } from '../redis/redis.service';

type EventEmitter = Server | Socket;

@WebSocketGateway({ namespace: '/api/calc' })
export class CalcGateway implements OnGatewayConnection {
  @WebSocketServer()
  server: Server;

  constructor(
    private jwtService: JwtService,
    private whitelistService: WhitelistService,
    private redisService: RedisService,
    @InjectModel(Calc.name) private calcModel: Model<Calc>
  ) { }

  async handleConnection(client: Socket) {
    const authToken = client.handshake.query?.token;
    if (!authToken || !(typeof authToken === 'string')) {
      client.emit('error', {
        message: 'Authorization header missing or invalid',
      });
      client.disconnect(true);
      return;
    }

    try {
      const payload = await this.jwtService.verifyAsync(authToken);

      if (!(await this.whitelistService.has(payload.jti))) {
        client.emit('error', { message: 'Token revoked' });
        client.disconnect(true);
        return;
      }

      client.emit('user', { message: `${payload.userId}` });
    } catch (e) {
      client.emit('error', { message: 'Invalid or expired token' });
      client.disconnect(true);
    }
  }

  private jobToSocketMap = new Map<string, string>();

  sendToSocket(socketId: string, event: string, data: any) {
    this.server.to(socketId).emit(event, data);
  }

  private sendJsonEvent(emitter: EventEmitter, event: string, data: any): void {
    const payload = typeof data === 'object' && data !== null ? data : { data };
    emitter.emit(event, JSON.stringify(payload));
  }

  @SubscribeMessage('calculate')
  async handleCalculate(
    @MessageBody() rawPayload: any,
    @ConnectedSocket() client: Socket
  ) {
    let userId = client.handshake.query?.userid;
    console.log(userId);
    if (!userId) {
      this.sendJsonEvent(client, 'error', { message: 'Unauthorized' });
      return;
    }
    if (Array.isArray(userId)) {
      userId = userId[1];
      console.log("USER ID IS ARRAY!!!", userId);
    }

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
        userId: userId,
        status: 'queued',
        observations: dto.observations,
      });
      await job.save();
      const jobId = job._id.toString();

      await this.redisService.addToStream('calculation_jobs', {
        jobId,
        userId: userId,
        socketId: client.id,
        observations: JSON.stringify(dto.observations),
      });

      this.sendJsonEvent(client, 'status', { jobId, status: 'queued' });
    } catch (error) {
      console.error('[CalcGateway] Failed to create job:', error);
      this.sendJsonEvent(client, 'error', {
        message: 'Failed to create calculation job',
      });
    }
  }
}
