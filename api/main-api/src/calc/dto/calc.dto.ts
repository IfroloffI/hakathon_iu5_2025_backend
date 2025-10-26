import {
  IsNumber,
  IsArray,
  Min,
  ArrayMinSize,
  ValidateNested,
  IsOptional,
} from 'class-validator';
import { Type } from 'class-transformer';

export class ObservationDto {
  // === Экваториальные координаты
  @IsNumber({}, { message: 'ra_hours must be a number' })
  ra_hours: number;

  @IsNumber({}, { message: 'dec_degrees must be a number' })
  dec_degrees: number;

  // === Время наблюдения
  @IsNumber({}, { message: 'timestamp must be a number (Unix seconds)' })
  timestamp: number;

  // === Горизонтальные координаты (альтернатива RA/Dec)
  @IsOptional()
  @IsNumber({}, { message: 'alt_degrees must be a number' })
  alt_degrees?: number;

  @IsOptional()
  @IsNumber({}, { message: 'az_degrees must be a number' })
  az_degrees?: number;

  // === Место наблюдения
  @IsOptional()
  @IsNumber({}, { message: 'observer_lat_deg must be a number' })
  observer_lat_deg?: number;

  @IsOptional()
  @IsNumber({}, { message: 'observer_lon_deg must be a number' })
  observer_lon_deg?: number;

  @IsOptional()
  @IsNumber({}, { message: 'observer_height_m must be a number' })
  observer_height_m?: number;

  // === Точность измерения
  @IsOptional()
  @IsNumber({}, { message: 'uncertainty_arcsec must be a number' })
  uncertainty_arcsec?: number;
}

export class CalcRequestDto {
  @IsArray({ message: 'observations must be an array' })
  @ArrayMinSize(5, { message: 'At least 5 observations required' })
  @ValidateNested({ each: true })
  @Type(() => ObservationDto)
  observations: ObservationDto[];

  @IsOptional()
  @IsNumber({}, { message: 'days_ahead must be a number' })
  days_ahead?: number;
}
