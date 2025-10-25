import {
  Body,
  Controller,
  Post,
  UseGuards,
  Get,
  Req,
  UnauthorizedException,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import { LoginDto } from './dto/login.dto';
import { RegisterDto } from './dto/register.dto';
import { AuthGuard } from '@nestjs/passport';
import { RequestWithUser } from './types';

@Controller('api/auth')
export class AuthController {
  constructor(private readonly authService: AuthService) { }

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
    return {
      id: req.user.userId,
      email: req.user.email,
      username: req.user.username,
    };
  }

  @UseGuards(AuthGuard('jwt'))
  @Post('logout')
  logout(@Req() req: RequestWithUser) {
    const jti = req.user.jti;
    this.authService.logout(jti);
    return { message: 'Logged out' };
  }
}
