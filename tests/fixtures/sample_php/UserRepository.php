<?php

namespace App\Models;

class UserRepository
{
    public function find(int $id): ?User
    {
        return User::where('id', $id)->first();
    }

    public function findByEmail(string $email): ?User
    {
        return User::where('email', $email)->first();
    }

    protected function buildQuery(): void
    {
        // Complex query builder
    }
}

function helperFunction(): string
{
    return 'helper';
}
