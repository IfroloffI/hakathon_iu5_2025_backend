import { Injectable } from '@nestjs/common';
import { CalcGateway } from './calc.gateway';

@Injectable()
export class NotificationService {
  constructor(private calcGateway: CalcGateway) {}

  emitCalcUpdate(socketId: string, event: string, data: any) {
    const payload = typeof data === 'object' && data !== null ? data : { data };
    this.calcGateway.server.to(socketId).emit(event, JSON.stringify(payload));
  }
}
