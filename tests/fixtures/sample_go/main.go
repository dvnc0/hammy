package main

import (
	"fmt"
	"net/http"
)

type UserService interface {
	FindByID(id string) (*User, error)
	Create(user *User) error
}

type User struct {
	ID    string
	Name  string
	Email string
}

func NewUser(name string, email string) *User {
	return &User{Name: name, Email: email}
}

func (u *User) FullName() string {
	return u.Name
}

func (u *User) Validate() error {
	if u.Name == "" {
		return fmt.Errorf("name required")
	}
	return nil
}

func fetchData(url string) {
	resp, _ := http.Get("http://api.example.com/users")
	defer resp.Body.Close()
}
