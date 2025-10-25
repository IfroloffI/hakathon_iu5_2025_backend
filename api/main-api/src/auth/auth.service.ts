import {
  Injectable,
  UnauthorizedException,
  BadRequestException,
  ConflictException,
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

  async findById(id: string) {
    return this.userModel.findById(id).select('-password');
  }

  async findUserById(userId: string) {
    return this.userModel.findById(userId).select('+email').exec();
  }

  async register(dto: RegisterDto): Promise<{ access_token: string }> {
    const existingUser = await this.userModel.findOne({
      $or: [{ email: dto.email }, { username: dto.username }],
    });

    if (existingUser) {
      if (existingUser.email === dto.email) {
        throw new ConflictException('Email already registered');
      }
      if (existingUser.username === dto.username) {
        throw new ConflictException('Username already taken');
      }
    }

    const hashedPassword = await bcrypt.hash(dto.password, 10);
    const user = await this.userModel.create({
      email: dto.email,
      username: dto.username,
      password: hashedPassword,
    });

    const savedUser = await this.userModel.findById(user._id);

    const { token, jti } = this.generateToken(user);
    this.whitelistService.add(jti);
    return { access_token: token };
  }

  async login(dto: LoginDto): Promise<{ access_token: string }> {
    const user = await this.userModel.findOne({
      $or: [{ email: dto.login }, { username: dto.login }],
    });

    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isMatch = await bcrypt.compare(dto.password, user.password);
    if (!isMatch) {
      throw new UnauthorizedException('Invalid email or password');
    }

    const { token, jti } = this.generateToken(user);
    this.whitelistService.add(jti);
    return { access_token: token };
  }

  logout(jti: string): void {
    this.whitelistService.del(jti);
  }

  isTokenValid(jti: string): boolean {
    return this.whitelistService.has(jti);
  }

  private generateToken(user: UserDocument): { token: string; jti: string } {
    const jti = crypto.randomUUID();

    const payload = {
      userId: user._id.toString(),
      jti,
      username: user.username,
      email: user.email,
    };

    const token = this.jwtService.sign(payload);
    return { token, jti };
  }
}
