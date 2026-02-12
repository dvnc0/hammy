import { fetchUsers } from "./api";

export class UserComponent {
    async loadUsers() {
        const users = await fetchUsers();
        this.render(users);
    }

    render(data) {
        return data.map(u => u.name);
    }
}

export function formatUser(user) {
    return `${user.firstName} ${user.lastName}`;
}
