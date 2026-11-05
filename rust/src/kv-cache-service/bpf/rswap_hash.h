/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * On-arena ABI shared by userspace and the rswap digest program.
 *
 * Userspace is the only writer. It publishes a page by storing the hash with
 * `ready == 0`, then storing `ready == 1`; the program reads `ready` first and
 * copies the hash only when it is non-zero.
 */
#ifndef RSWAP_HASH_H
#define RSWAP_HASH_H

/* Kernel context: the program fills @hash and returns non-zero on success. */
struct rswap_hash_ctx {
	unsigned long long vaddr;	/* in: base-page address in vma->vm_mm */
	unsigned char hash[48];		/* out: 48-byte digest */
};

#define RSWAP_ARENA_MAGIC	0x5253574150415248ULL	/* "RSWAPARH" */
#define RSWAP_ARENA_VERSION	1ULL
#define RSWAP_ARENA_ENTRIES_OFF	4096ULL
#define RSWAP_ARENA_ENTRY_SIZE	64ULL

/* Arena page 0: geometry and the location of the entry table. */
struct rswap_arena_hdr {
	unsigned long long magic;
	unsigned long long version;
	unsigned long long page_shift;
	unsigned long long data_base_va;
	unsigned long long entries_off;
	unsigned long long entry_size;
	unsigned long long num_entries;
	unsigned long long pages_per_chunk;
	unsigned long long reserved[8];
};

/* One entry per base page of the service's shared mapping. */
struct rswap_page_meta {
	unsigned long long ready;	/* 1 = hash valid, written last */
	unsigned char hash[48];
	unsigned long long page_index;	/* absolute base-page index */
};

#endif /* RSWAP_HASH_H */
