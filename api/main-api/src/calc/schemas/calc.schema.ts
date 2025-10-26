import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { HydratedDocument } from 'mongoose';

export type CalcDocument = HydratedDocument<Calc>;

@Schema({ timestamps: true })
export class Calc {
  @Prop({ required: true, index: true })
  userId: string;

  @Prop({
    required: true,
    enum: ['queued', 'processing', 'completed', 'failed', 'deleted'],
    default: 'queued',
    index: true,
  })
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'deleted';

  // --- Наблюдения ---
  @Prop({ required: true })
  observations: {
    ra_hours: number;
    dec_degrees: number;
    timestamp: number;
    // Горизонтальные координаты (альтернатива RA/Dec)
    alt_degrees?: number;
    az_degrees?: number;
    // Место наблюдения
    observer_lat_deg?: number;
    observer_lon_deg?: number;
    observer_height_m?: number;
    // Точность измерения
    uncertainty_arcsec?: number;
  }[];

  // --- Параметры запроса ---
  @Prop()
  days_ahead?: number;

  // --- Результаты ---
  @Prop()
  success?: boolean;

  @Prop()
  error?: string;

  @Prop()
  semi_major_axis_au?: number;
  @Prop()
  eccentricity?: number;
  @Prop()
  inclination_deg?: number;
  @Prop()
  longitude_ascending_node_deg?: number;
  @Prop()
  argument_perihelion_deg?: number;
  @Prop()
  perihelion_passage_jd?: number;

  // --- Метрики качества ---
  @Prop()
  rms_residual_deg?: number;

  // --- Сближение с Землёй ---
  @Prop()
  closest_approach_jd?: number;
  @Prop()
  closest_distance_au?: number;

  createdAt: Date;
  updatedAt: Date;
}

export const CalcSchema = SchemaFactory.createForClass(Calc);

CalcSchema.index({ userId: 1, status: 1 });
