import {
  Body,
  Controller,
  Post,
  UseGuards,
  Get,
  Req,
  Res,
  HttpStatus,
  UnauthorizedException,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import { LoginDto } from './dto/login.dto';
import { RegisterDto } from './dto/register.dto';
import { AuthGuard } from '@nestjs/passport';
import { Request, Response } from 'express';
import { RequestWithUser } from './types';

@Controller('api/auth')
export class AuthController {
  constructor(private readonly authService: AuthService) {}

  @Post('register')
  async register(@Body() dto: RegisterDto) {
    return this.authService.register(dto);
  }

  @Post('login')
  async login(@Body() dto: LoginDto) {
    return this.authService.login(dto);
  }

  @UseGuards(AuthGuard('jwt'))
  @Get('check')
  async check(@Req() req: RequestWithUser) {
    const user = await this.authService.findUserById(req.user.userId);
    if (!user) {
      throw new UnauthorizedException('User not found');
    }

    return {
      id: user._id.toHexString(),
      email: user.email,
    };
  }

  @UseGuards(AuthGuard('jwt'))
  @Post('logout')
  logout(@Req() req: RequestWithUser) {
    const jti = req.user['jti'];
    if (typeof jti !== 'string') {
      throw new UnauthorizedException('Invalid session');
    }
    this.authService.logout(jti);
    return { message: 'Logged out' };
  }
}
