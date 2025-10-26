import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { RedisService } from '../redis/redis.service';
import { CalcService } from '../calc/calc.service';
import { NotificationService } from '../calc/notification.service';

@Injectable()
export class CalcWorkerService implements OnModuleInit {
  private readonly logger = new Logger(CalcWorkerService.name);
  private client;
  private readonly streamKey = 'calculation_jobs';
  private readonly groupName = 'calc-workers';
  private readonly consumerName = `worker-${Math.random().toString(36).slice(2, 8)}`;

  constructor(
    private readonly redisService: RedisService,
    private readonly calcService: CalcService,
    private readonly notificationService: NotificationService
  ) {}

  async onModuleInit() {
    this.client = await this.redisService.getClient();

    await this.ensureGroupExists();
    this.logger.log(`Calc worker "${this.consumerName}" started`);
    this.consumeJobs();
  }

  private async ensureGroupExists() {
    try {
      await this.client.xGroupCreate(this.streamKey, this.groupName, '0', {
        MKSTREAM: true,
      });
      this.logger.log(`Created consumer group "${this.groupName}"`);
    } catch (err: any) {
      if (err.message.includes('BUSYGROUP')) {
        this.logger.log(`Consumer group "${this.groupName}" already exists`);
      } else {
        this.logger.error('Error creating consumer group:', err);
      }
    }
  }

  private async consumeJobs() {
    while (true) {
      try {
        const messages = await this.client.xReadGroup(
          this.groupName,
          this.consumerName,
          [{ key: this.streamKey, id: '>' }],
          { COUNT: 1, BLOCK: 5000 }
        );

        if (!messages) continue; // нет задач — ждём дальше

        for (const stream of messages) {
          for (const message of stream.messages) {
            const { id, message: fields } = message;
            await this.processJob(id, fields);
          }
        }
      } catch (err) {
        this.logger.error('Error consuming jobs:', err);
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
  }

  private async processJob(id: string, fields: Record<string, string>) {
    const jobId = fields.jobId;
    const socketId = fields.socketId;
    const obs = JSON.parse(fields.obs || '[]');

    this.logger.log(`Processing job ${jobId}`);

    try {
      await this.calcService.updateCalc(jobId, { status: 'processing' });
      this.notificationService.emitCalcUpdate(socketId, 'status', {
        jobId,
        status: 'processing',
      });

      const result = await this.calcService.calculateOrbit(obs);

      await this.calcService.updateCalc(jobId, {
        status: 'completed',
        ...(typeof result === 'object' ? result : { raw: String(result) }),
      });

      this.notificationService.emitCalcUpdate(socketId, 'result', {
        jobId,
        ...(typeof result === 'object' ? result : { raw: String(result) }),
      });

      await this.client.xAck(this.streamKey, this.groupName, id);
      this.logger.log(`Job ${jobId} processed and acknowledged`);
    } catch (err) {
      this.logger.error(`Failed to process job ${jobId}:`, err);
    }
  }
}
