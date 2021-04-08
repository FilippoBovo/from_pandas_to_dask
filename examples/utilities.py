def ordinal(n):
    """Get the ordinal format of an integer number."""
    # Source: https://stackoverflow.com/a/20007730
    return "{:d}{:s}".format(
        n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4]
    )


def print_ddf(ddf, end="\n"):
    table_rows = ddf.compute().to_string().split("\n")
    max_length = max([len(row) for row in table_rows])
    if ddf.index.name is None:
        header = table_rows[0]
    else:
        header = "\n".join(table_rows[:2])

    print(header)
    for i in range(ddf.npartitions):
        print("-" * max_length + f" {ordinal(i + 1)} partition")
        partition_df = ddf.get_partition(i).compute()
        if len(partition_df) > 0:
            if partition_df.index.name is None:
                print("\n".join(partition_df.to_string().split("\n")[1:]))
            else:
                print("\n".join(partition_df.to_string().split("\n")[2:]))
    print(end, end="")
