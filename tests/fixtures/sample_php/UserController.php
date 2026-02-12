<?php

namespace App\Controllers;

use App\Models\User;
use App\Services\PaymentService;

#[Route('/api/v1/users')]
class UserController
{
    public function getUser(int $id): User
    {
        return User::find($id);
    }

    #[Route('/api/v1/users/{id}/pay')]
    public function processPayment(int $id, float $amount): void
    {
        $service = new PaymentService();
        $service->charge($id, $amount);
    }

    private function validateInput(array $data): bool
    {
        return !empty($data);
    }
}
