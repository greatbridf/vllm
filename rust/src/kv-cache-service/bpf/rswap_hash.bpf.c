/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * rswap digest program: read one page's digest out of arena metadata.
 *
 * The kernel calls this with {vaddr, hash[48]} under the page table lock, so
 * it must not sleep and must not fault. The arena base is a load-time constant:
 * the userspace loader replaces the RSWAP_ARENA_MARKER immediate with the
 * arena's user VA. With BPF_F_ARENA_SCALAR every load through that scalar is
 * rewritten to a BPF_PROBE_MEM32 arena access, so the program needs no map,
 * no helpers and no relocations.
 */
#include "rswap_hash.h"

typedef unsigned long long u64;

#define SEC(name) __attribute__((section(name), used))

/* Patched by the loader; must stay a single ld_imm64 in the program. */
#define RSWAP_ARENA_MARKER 0x5253574150524f42ULL /* "RSWAPROB" */

_Static_assert(sizeof(struct rswap_hash_ctx) == 56, "rswap context ABI");
_Static_assert(sizeof(struct rswap_arena_hdr) == 128, "arena header ABI");
_Static_assert(sizeof(struct rswap_page_meta) == 64, "arena entry ABI");
_Static_assert(__builtin_offsetof(struct rswap_page_meta, hash) == 8, "hash ABI");
_Static_assert(__builtin_offsetof(struct rswap_page_meta, page_index) == 56,
	       "page index ABI");

SEC("syscall")
int rswap_hash_prog(struct rswap_hash_ctx *ctx)
{
	u64 base = RSWAP_ARENA_MARKER;
	struct rswap_arena_hdr *hdr;
	volatile struct rswap_page_meta *meta;
	u64 vaddr = ctx->vaddr;
	u64 shift, data_base, num, index, entry_size;
	u64 *dst;
	int i;

	/* Keep the base opaque so offsets stay register-relative and the marker
	 * remains a single ld_imm64 for the loader to replace. */
	asm volatile("" : "+r"(base));
	hdr = (struct rswap_arena_hdr *)base;

	if (*(volatile u64 *)&hdr->magic != RSWAP_ARENA_MAGIC ||
	    *(volatile u64 *)&hdr->version != RSWAP_ARENA_VERSION)
		return 0;

	entry_size = *(volatile u64 *)&hdr->entry_size;
	shift = *(volatile u64 *)&hdr->page_shift;
	data_base = *(volatile u64 *)&hdr->data_base_va;
	num = *(volatile u64 *)&hdr->num_entries;

	if (entry_size != RSWAP_ARENA_ENTRY_SIZE || shift >= 32 ||
	    vaddr < data_base)
		return 0;

	index = (vaddr - data_base) >> shift;
	if (index >= num)
		return 0;

	meta = (volatile struct rswap_page_meta *)(base +
		*(volatile u64 *)&hdr->entries_off +
		index * RSWAP_ARENA_ENTRY_SIZE);
	if (meta->ready == 0)
		return 0;

	dst = (u64 *)ctx->hash;
#pragma clang loop unroll(full)
	for (i = 0; i < 6; i++)
		dst[i] = meta->hash_words[i];

	return 1;
}
