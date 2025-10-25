import {
  Injectable,
  UnauthorizedException,
  BadRequestException,
} from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import * as bcrypt from 'bcrypt';
import { JwtService } from '@nestjs/jwt';
import * as crypto from 'crypto';
import { User, UserDocument } from './schemas/user.schema';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
import { WhitelistService } from './whitelist.service';

@Injectable()
export class AuthService {
  constructor(
    @InjectModel(User.name) private userModel: Model<UserDocument>,
    private jwtService: JwtService,
    private whitelistService: WhitelistService
  ) {}

  async findUserById(userId: string) {
    return this.userModel.findById(userId).select('+email').exec();
  }

  async register(dto: RegisterDto): Promise<{ access_token: string }> {
    const existingUser = await this.userModel.findOne({ email: dto.email });
    if (existingUser) {
      throw new BadRequestException('Email already registered');
    }

    const hashedPassword = await bcrypt.hash(dto.password, 10);
    const user = await this.userModel.create({
      email: dto.email,
      password: hashedPassword,
    });

    const { token, jti } = this.generateToken(user._id.toString());
    this.whitelistService.add(jti);
    return { access_token: token };
  }

  async login(dto: LoginDto): Promise<{ access_token: string }> {
    const user = await this.userModel.findOne({ email: dto.email });
    if (!user) {
      throw new UnauthorizedException('Invalid email or password');
    }

    const isMatch = await bcrypt.compare(dto.password, user.password);
    if (!isMatch) {
      throw new UnauthorizedException('Invalid email or password');
    }

    const { token, jti } = this.generateToken(user._id.toString());
    this.whitelistService.add(jti);
    return { access_token: token };
  }

  logout(jti: string): void {
    this.whitelistService.del(jti);
  }

  isTokenValid(jti: string): boolean {
    return this.whitelistService.has(jti);
  }

  private generateToken(userId: string): { token: string; jti: string } {
    const jti = crypto.randomUUID();
    const payload = { userId, jti };
    const token = this.jwtService.sign(payload);
    return { token, jti };
  }
}
