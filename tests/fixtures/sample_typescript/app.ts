import { Request, Response } from 'express';
import { UserService } from './services';

interface UserDTO {
    id: number;
    name: string;
    email: string;
}

enum UserRole {
    Admin = "ADMIN",
    User = "USER",
    Guest = "GUEST",
}

class UserController {
    private service: UserService;

    constructor(service: UserService) {
        this.service = service;
    }

    public async getUser(req: Request, res: Response): Promise<UserDTO> {
        const user = await this.service.findById(req.params.id);
        return user;
    }

    public async createUser(req: Request, res: Response): void {
        await this.service.create(req.body);
        res.status(201).send();
    }

    private validateEmail(email: string): boolean {
        return email.includes('@');
    }
}

export const fetchUsers = async (): Promise<UserDTO[]> => {
    const response = await fetch('/api/v1/users');
    return response.json();
};

export function formatUser(user: UserDTO): string {
    return `${user.name} <${user.email}>`;
}
