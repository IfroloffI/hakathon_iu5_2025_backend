import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { Calc, CalcDocument } from './schemas/calc.schema';

const PROTO_PATH = __dirname + '/assets/proto/calc.proto';

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

const calcProto: any = grpc.loadPackageDefinition(packageDefinition).calc;

@Injectable()
export class CalcService {
  private client: any;

  constructor(@InjectModel(Calc.name) private calcModel: Model<CalcDocument>) {
    const host = process.env.CALC_API_HOST || 'localhost:50051';
    this.client = new calcProto.CometCalculator(
      host,
      grpc.credentials.createInsecure()
    );
  }

  async calculateOrbitOld(
    observations: { ra_hours: number; dec_degrees: number; timestamp: number }[]
  ) {
    return new Promise((resolve) => {
      this.client.CalculateOrbit({ observations }, (err, response) => {
        if (err) {
          resolve({ success: false, error: err.message });
        } else {
          resolve(response);
        }
      });
    });
  }

  async calculateOrbit(observations: any[], days_ahead?: number): Promise<Record<string, any>> {
    return new Promise((resolve) => {
      this.client.CalculateOrbit({ 
        observations, 
        days_ahead: days_ahead || 1460 // 4 года по умолчанию
      }, (err, response) => {
        if (err) {
          resolve({ success: false, error: err.message });
        } else {
          resolve(response || { success: false, error: 'Empty response' });
        }
      });
    });
  }

  async findAllByUserId(userId: string): Promise<Calc[]> {
    return this.calcModel
      .find({
        status: { $ne: 'deleted' },
      })
      .sort({ updatedAt: -1 })
      .lean()
      .exec();
  }

  async findByIdAndUserId(
    id: string,
    userId: string
  ): Promise<Calc & { id: string }> {
    const doc = await this.calcModel
      .findOne({
        _id: id,
        userId,
        status: { $ne: 'deleted' },
      })
      .lean()
      .exec();

    if (!doc) {
      throw new NotFoundException('Calc record not found or already deleted');
    }

    const { _id, ...rest } = doc;
    return { ...rest, id: _id.toString() };
  }

  async softDeleteByIdAndUserId(id: string, userId: string): Promise<void> {
    const result = await this.calcModel
      .updateOne(
        {
          _id: id,
          userId,
          status: { $ne: 'deleted' },
        },
        {
          $set: { status: 'deleted' },
        }
      )
      .exec();

    if (result.matchedCount === 0) {
      throw new NotFoundException('Calc record not found or already deleted');
    }
  }

  async updateCalc(jobId: string, update: Record<string, any>) {
    try {
      const updated = await this.calcModel.findByIdAndUpdate(jobId, update, {
        new: true,
      });
      if (!updated) {
        throw new NotFoundException('Job record not found');
      }
      return updated;
    } catch (err) {
      throw err;
    }
  }
}
