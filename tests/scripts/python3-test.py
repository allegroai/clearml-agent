def main():
    if not (1 / 2 == 0.5):
        raise ValueError('failure')
    print('success')


if __name__ == "__main__":
    main()
