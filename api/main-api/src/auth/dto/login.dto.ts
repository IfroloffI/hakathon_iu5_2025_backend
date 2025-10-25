import { IsString, IsNotEmpty, MinLength, IsOptional } from 'class-validator';

export class LoginDto {
  @IsString()
  @IsNotEmpty()
  login: string; // Может быть email или username

  @IsString()
  @IsNotEmpty()
  @MinLength(6)
  password: string;
}
