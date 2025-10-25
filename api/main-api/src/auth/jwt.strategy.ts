import { Injectable, UnauthorizedException } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy, StrategyOptions } from 'passport-jwt';
import { ConfigService } from '@nestjs/config';
import { WhitelistService } from './whitelist.service';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(
    private readonly whitelistService: WhitelistService,
    private readonly configService: ConfigService
  ) {
    const jwtSecret = configService.get<string>(
      'JWT_SECRET',
      'fallback-secret-for-dev'
    );

    if (!jwtSecret || jwtSecret === 'fallback-secret-for-dev') {
      throw new Error('JWT_SECRET is not configured');
    }

    const options: StrategyOptions = {
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: jwtSecret,
    };

    super(options);
  }

  async validate(payload: any): Promise<{
    userId: string;
    jti: string;
    username: string;
    email: string;
  }> {
    if (!payload) {
      throw new UnauthorizedException('Invalid token payload');
    }

    const { userId, jti, username, email } = payload;

    if (!userId || typeof userId !== 'string') {
      throw new UnauthorizedException('Token missing or invalid userId');
    }

    if (!jti || typeof jti !== 'string') {
      throw new UnauthorizedException('Token missing or invalid jti');
    }

    if (!username || typeof username !== 'string') {
      throw new UnauthorizedException('Token missing or invalid username');
    }

    if (!email || typeof email !== 'string') {
      throw new UnauthorizedException('Token missing or invalid email');
    }

    if (!this.whitelistService.has(jti)) {
      throw new UnauthorizedException('Token revoked or expired');
    }

    return { userId, jti, username, email };
  }
}
