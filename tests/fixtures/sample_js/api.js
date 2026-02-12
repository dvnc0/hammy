import { config } from "./config";

export async function fetchUsers() {
    const response = await fetch("/api/v1/users");
    return response.json();
}

export async function fetchUserProfile(userId) {
    const response = await fetch(`/api/v1/users/${userId}`);
    return response.json();
}

const submitPayment = async (userId, amount) => {
    await axios.post("/api/v1/users/{id}/pay", { userId, amount });
};
