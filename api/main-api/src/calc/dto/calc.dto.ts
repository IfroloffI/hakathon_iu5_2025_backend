import {
  IsNumber,
  IsArray,
  Min,
  ArrayMinSize,
  ValidateNested,
} from 'class-validator';
import { Type } from 'class-transformer';

export class ObservationDto {
  @IsNumber({}, { message: 'ra_hours must be a number' })
  ra_hours: number;

  @IsNumber({}, { message: 'dec_degrees must be a number' })
  dec_degrees: number;

  @IsNumber({}, { message: 'timestamp must be a number (Unix seconds)' })
  timestamp: number;
}

export class CalcRequestDto {
  @IsArray({ message: 'observations must be an array' })
  @ArrayMinSize(5, { message: 'At least 5 observations required' })
  @ValidateNested({ each: true })
  @Type(() => ObservationDto)
  observations: ObservationDto[];
}
