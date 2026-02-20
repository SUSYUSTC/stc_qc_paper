from .stc import Diagram
'''
minimal: apply STC to all N^6 and part of N^5 diagrams, leave others for exact evaluatino
full: apply STC to all N^6 and all N^5 diagrams
'''

indices_occ = 'ijkl'
indices_vir = 'abcd'
indices_aux = 'x'

preevaluate_X1 = True
reorder_Hoovv = True


def Hoovv_to_Hovov(diagram: Diagram):
    # reorder pqrs to prqs
    if 'Hoovv' in diagram.symbols:
        index = diagram.symbols.index('Hoovv')
        expr_in, expr_out = diagram.expr.split('->')
        exprs_in = expr_in.split(',')
        o1, o2, v1, v2 = exprs_in[index]
        exprs_in[index] = f'{o1}{v1}{o2}{v2}'
        diagram.expr = '->'.join([','.join(exprs_in), expr_out])
    return diagram


# we don't preevaluate X2 to avoid excessive memory use

intermediate_tensors = [
    ('Rov_X1_full',   [Diagram('iax,ia->x',   ('Rov', 'X1'),           +1.0), ]),
    ('Rov_X1_oo',     [Diagram('iax,ja->ijx', ('Rov', 'X1'),           +1.0), ]),
    ('Rvv_X1_ov',     [Diagram('abx,ia->ibx', ('Rvv', 'X1'),           +1.0), ]),
    ('Rov_X1X1_ov',   [Diagram('ijx,ib->jbx', ('Rov_X1_oo', 'X1'),     +1.0), ]),
    ('Roo_X1_ov',     [Diagram('ijx,ja->iax', ('Roo', 'X1'),           +1.0), ]),
    ('Rov_part_oo',   [Diagram('iax->iax',    ('Roo_X1_ov', ),         +1.0),
                       Diagram('iax->iax',    ('Rov', ),               -2.0),
                       Diagram('iax->iax',    ('Rov_X1X1_ov', ),       +2.0), ]),
    ('Rov_part_vv',   [Diagram('iax->iax',    ('Rvv_X1_ov', ),         +1.0),
                       Diagram('iax->iax',    ('Rov', ),               +2.0),
                       Diagram('iax->iax',    ('Rov_X1X1_ov', ),       -2.0), ]),
    ('Hovov_symm_X1', [Diagram('iax,x->ia',   ('Rov', 'Rov_X1_full'),  +2.0), 
                       Diagram('iajb,ib->ja', ('Hovov', 'X1'),         -1.0), ]),
    ('Loo_X1',        [Diagram('x,kix->ki',   ('Rov_X1_full', 'Roo'),  +2.0), 
                       Diagram('klx,ilx->ki', ('Rov_X1_oo', 'Roo'),    -1.0), ]),
    ('Loo_X1X1',      [Diagram('kc,ic->ki',   ('Hovov_symm_X1', 'X1'), +1.0), ]),
    ('Lvv_X1',        [Diagram('x,acx->ac',   ('Rov_X1_full', 'Rvv'),  -2.0), 
                       Diagram('ckx,kax->ac', ('Rvo', 'Rvv_X1_ov'),    +1.0), ]),
    ('Lvv_X1X1',      [Diagram('kc,ka->ac',   ('Hovov_symm_X1', 'X1'), +1.0), ]),
    ('Loo',           [Diagram('ij->ij',      ('Loo_X1', ),            +1.0),
                       Diagram('ij->ij',      ('Loo_X1X1', ),          +1.0), 
                       Diagram('ij->ij',      ('Focc', ),              +1.0), ]),
    ('Lvv',           [Diagram('ab->ab',      ('Lvv_X1', ),            +1.0),
                       Diagram('ab->ab',      ('Lvv_X1X1', ),          +1.0), 
                       Diagram('ab->ab',      ('Fvir', ),              -1.0), ]),
    ('Lov',           [Diagram('kcx,x->kc',   ('Rov', 'Rov_X1_full'),  +2.0), 
                       Diagram('kdlc,ld->kc', ('Hovov', 'X1'),         -1.0), ]),
    ('Lov_X1_oo',     [Diagram('kc,ic->ki',   ('Lov', 'X1'),           +1.0), ]),
    #('X2_symm',       [Diagram('iajb->iajb',  ('X2', ),                +2.0),
    #                   Diagram('iajb->jaib',  ('X2', ),                -1.0), ]),
    #('X2_vvoo',       [Diagram('iajb->abij',  ('X2', ),                +1.0), ]),
]

intermediate_tensors_minimal_add = [
    ('XRov_symm', [Diagram('iajb,jbx->iax',   ('X2_symm', 'Rov'),      +1.0), ]),
    ('Rov_full',  [Diagram('iax->iax',        ('Rov', ),               +2.0), 
                   Diagram('iax->iax',        ('XRov_symm', ),         +1.0),
                   Diagram('iax->iax',        ('Rov_X1X1_ov', ),       -2.0),
                   Diagram('iax->iax',        ('Rvv_X1_ov', ),         +2.0),
                   Diagram('iax->iax',        ('Roo_X1_ov', ),         -2.0), ]),
    ('Loo_X2',    [Diagram('kcx,icx->ki',     ('Rov', 'XRov_symm'),    +1.0), ]),
    ('Lvv_X2',    [Diagram('ckx,kax->ac',     ('Rvo', 'XRov_symm'),    +1.0), ]),
    ('Loo_full',  [Diagram('ij->ij',          ('Loo_X1', ),            +1.0),
                   Diagram('ij->ij',          ('Loo_X2', ),            +1.0),
                   Diagram('ij->ij',          ('Loo_X1X1', ),          +1.0), 
                   Diagram('ij->ij',          ('Focc', ),              +1.0), ]),
    ('Lvv_full',  [Diagram('ab->ab',          ('Lvv_X1', ),            +1.0),
                   Diagram('ab->ab',          ('Lvv_X2', ),            +1.0),
                   Diagram('ab->ab',          ('Lvv_X1X1', ),          +1.0), 
                   Diagram('ab->ab',          ('Fvir', ),              -1.0), ]),
]

CCSD_X1_common = [
    Diagram('ia,->ia',      ('X1', 'madelung'),           -1.0),

    Diagram('ki,ka->ia',    ('Loo_X1X1', 'X1'),           -1.0),
    Diagram('ac,ic->ia',    ('Lvv_X1X1', 'X1'),           -1.0),
    Diagram('ki,ka->ia',    ('Focc', 'X1'),               -1.0),
    Diagram('ac,ic->ia',    ('Fvir', 'X1'),               +1.0),

    Diagram('iajb,jb->ia',  ('Hovov', 'X1'),              +2.0),
    Diagram('ijab,jb->ia',  ('Hoovv', 'X1'),              -1.0),

    Diagram('kc,iakc->ia',  ('Lov', 'X2_symm'),           +1.0),
    Diagram('ki,ka->ia',    ('Lov_X1_oo', 'X1'),          +1.0),

    Diagram('iax,x->ia',    ('Rvv_X1_ov', 'Rov_X1_full'), +2.0),
    Diagram('kix,kax->ia',  ('Rov_X1_oo', 'Rvv_X1_ov'),   -1.0),
    Diagram('iax,x->ia',    ('Roo_X1_ov', 'Rov_X1_full'), -2.0),
    Diagram('lax,lix->ia',  ('Rov_X1X1_ov', 'Roo'),       +1.0),
]

if reorder_Hoovv:
    CCSD_X1_common = [Hoovv_to_Hovov(diag) for diag in CCSD_X1_common]

CCSD_X1_common_nonDF = []

CCSD_X1_common_DF = []

CCSD_X1_full_common = [
    Diagram('kcld,icld,ka->ia', ('Hovov', 'X2_symm', 'X1'), -1.0, path=['k->cld', 'k->a', 'cld->i']),
    Diagram('kcld,kald,ic->ia', ('Hovov', 'X2_symm', 'X1'), -1.0, path=['c->kld', 'c->i', 'kld->a']),
    Diagram('lcki,kalc->ia',    ('Hovoo', 'X2_symm'),       -1.0, path=['lck->i', 'lck->a']),
]
CCSD_X1_full_nonDF = [
    Diagram('kdac,ickd->ia',    ('Hovvv', 'X2_symm'),       +1.0, path=['kdc->i', 'kdc->a']),
]
CCSD_X1_full_DF = [
    Diagram('kdx,acx,ickd->ia', ('Rov', 'Rvv', 'X2_symm'),  +1.0, path=['cx->kd', 'cx->a', 'ckd->i']),
]

CCSD_X1_minimal_common = [
    Diagram('ki,ka->ia',   ('Loo_X2', 'X1'),     -1.0),
    Diagram('ac,ic->ia',   ('Lvv_X2', 'X1'),     -1.0),
    Diagram('acx,icx->ia', ('Rvv', 'XRov_symm'), +1.0),
    Diagram('kix,kax->ia', ('Roo', 'XRov_symm'), -1.0),
]

CCSD_X1_minimal_nonDF = []

CCSD_X1_minimal_DF = []

if preevaluate_X1:
    intermediate_tensors.append(('X1_common', CCSD_X1_common))
    CCSD_X1_common = [Diagram('ia->ia', ('X1_common', ), +1.0)]
    intermediate_tensors_minimal_add.append(('X1_minimal_common', CCSD_X1_minimal_common))
    CCSD_X1_minimal_common = [Diagram('ia->ia', ('X1_minimal_common', ), +1.0)]

CCSD_X2_common = [
    Diagram('iajb->iajb',             ('Hovov', ),                       +1.0),
    Diagram('iajb,->iajb',            ('X2', 'madelung'),                -2.0),

    Diagram('kilj,kalb->iajb',        ('Hoooo', 'X2'),                   +1.0, path=['kl->ij', 'kl->ab']),

    Diagram('kica,kcjb->iajb',        ('Hoovv', 'X2'),                   -2.0, path=['kc->ia', 'kc->jb']),
    Diagram('kicb,jcka->iajb',        ('Hoovv', 'X2'),                   -2.0, path=['kc->ib', 'kc->ja']),

    # HovovdiffXoccXvir
    Diagram('kcld,kalb,cdij->iajb',   ('Hovov', 'X2', 'X2_vvoo'),        +1.0, path=['kl->cd', 'kl->ab', 'cd->ij']),
    Diagram('kcld,kalb,ic,jd->iajb',  ('Hovov', 'X2', 'X1', 'X1'),       +1.0, path=['kl->cd', 'kl->ab', 'c->i', 'd->j']),
    Diagram('kcld,ka,lb,cdij->iajb',  ('Hovov', 'X1', 'X1', 'X2_vvoo'),  +1.0, path=['kl->cd', 'cd->ij', 'k->a', 'l->b']),

    Diagram('kdlc,ldia,kcjb->iajb',   ('Hovov', 'X2', 'X2'),             -2.0, path=['ld->kc', 'ld->ia', 'kc->jb']), # HovovdiffXsameXsame
    Diagram('kdlc,ldia,jckb->iajb',   ('Hovov', 'X2', 'X2'),             +2.0, path=['ld->kc', 'ld->ia', 'kc->jb']), # HovovdiffXsameXdiff
    Diagram('kdlc,jdla,ickb->iajb',   ('Hovov', 'X2', 'X2'),             +1.0, path=['ld->kc', 'ld->ja', 'kc->ib']), # HovovdiffXdiffXdiff
    Diagram('kdlc,id,la,kcjb->iajb',  ('Hovov', 'X1', 'X1', 'X2'),       +2.0, path=['ld->kc', 'd->i', 'l->a', 'kc->jb']), # HovovdiffXsameX1X2
    Diagram('kdlc,jd,la,ickb->iajb',  ('Hovov', 'X1', 'X1', 'X2'),       +2.0, path=['ld->kc', 'd->j', 'l->a', 'kc->ib']), # HovovdiffXdiffX1X2
    Diagram('lcki,jc,kalb->iajb',     ('Hovoo', 'X1', 'X2'),             +2.0, path=['lck->i', 'c->j', 'kl->ab']), # Hoooo
    Diagram('lcki,la,kcjb->iajb',     ('Hovoo', 'X1', 'X2'),             +2.0, path=['lck->i', 'l->a', 'kc->jb']), # Hdiff
    Diagram('lcki,lb,jcka->iajb',     ('Hovoo', 'X1', 'X2'),             +2.0, path=['lck->i', 'l->b', 'kc->ja']) # Hdiff
]
CCSD_X2_common_nonDF = [
    Diagram('cadb,cdij->iajb',        ('Hvvvv', 'X2_vvoo'),              +1.0, path=['cd->ij', 'cd->ab']),
    Diagram('kdca,kb,cdij->iajb',     ('Hovvv', 'X1', 'X2_vvoo'),        -2.0, path=['cd->k', 'kcd->a', 'k->b', 'cd->ij']), # Hvvvv
    Diagram('kcda,ic,kdjb->iajb',     ('Hovvv', 'X1', 'X2'),             -2.0, path=['kd->c', 'kdc->a', 'c->i', 'kd->jb']), # Hdiff
    Diagram('kcda,jc,kbid->iajb',     ('Hovvv', 'X1', 'X2'),             -2.0, path=['kd->c', 'kdc->a', 'c->j', 'kd->ib']), # Hdiff

    Diagram('iacb,jc->iajb',          ('Hovvv', 'X1'),                   +2.0, path=['c->iab', 'c->j']),
    Diagram('cadb,ic,jd->iajb',       ('Hvvvv', 'X1', 'X1'),             +1.0, path=['cd->ab', 'c->i', 'd->j']),
    Diagram('kdca,kb,ic,jd->iajb',    ('Hovvv', 'X1', 'X1', 'X1'),       -2.0, path=['cd->k', 'kcd->a', 'k->b', 'c->i', 'd->j']), # Hvvvv

    Diagram('iakj,kb->iajb',          ('Hovoo', 'X1'),                   -2.0, path=['k->iaj', 'k->b']),
    Diagram('kilj,ka,lb->iajb',       ('Hoooo', 'X1', 'X1'),             +1.0, path=['kl->ij', 'k->a', 'l->b']),
    Diagram('lcki,jc,ka,lb->iajb',    ('Hovoo', 'X1', 'X1', 'X1'),       +2.0, path=['lck->i', 'c->j', 'k->a', 'l->b']), # Hoooo

    Diagram('kicb,ka,jc->iajb',       ('Hoovv', 'X1', 'X1'),             -2.0, path=['kc->ib', 'c->j', 'k->a']),
    Diagram('kcld,ka,lb,ic,jd->iajb', ('Hovov', 'X1', 'X1', 'X1', 'X1'), +1.0, path=['kl->cd', 'c->i', 'd->j', 'k->a', 'l->b']),
    Diagram('kcia,jc,kb->iajb',       ('Hovov', 'X1', 'X1'),             -2.0, path=['kc->ia', 'c->j', 'k->b']),
]

if reorder_Hoovv:
    CCSD_X2_common = [Hoovv_to_Hovov(diag) for diag in CCSD_X2_common]
    CCSD_X2_common_nonDF = [Hoovv_to_Hovov(diag) for diag in CCSD_X2_common_nonDF]

CCSD_X2_common_DF = [
    Diagram('cax,dbx,cdij->iajb',     ('Rvv', 'Rvv', 'X2_vvoo'),         +1.0, path=['cd->x', 'cx->a', 'dx->b', 'cd->ij']),

    Diagram('kdx,cax,kb,cdij->iajb',  ('Rov', 'Rvv', 'X1', 'X2_vvoo'),   -2.0, path=['cd->x', 'cx->a', 'dx->k', 'k->b', 'cd->ij']), # Hvvvv
    Diagram('kix,dax,kdjb->iajb',     ('Rov_X1_oo', 'Rvv', 'X2'),        -2.0, path=['kd->x', 'kx->i', 'dx->a', 'kd->jb']), # Hdiff
    Diagram('kjx,dax,idkb->iajb',     ('Rov_X1_oo', 'Rvv', 'X2'),        -2.0, path=['kd->x', 'kx->j', 'dx->a', 'kd->ib']), # Hdiff

    Diagram('iax,jbx->iajb',          ('Rvv_X1_ov', 'Rov_part_vv'),      +1.0, path=['x->ia', 'x->jb']),
    #Diagram('iax,jbx->iajb',          ('Rvv_X1_ov', 'Rov'),              +2.0, path=['x->ia', 'x->jb']),
    #Diagram('iax,jbx->iajb',          ('Rvv_X1_ov', 'Rvv_X1_ov'),        +1.0, path=['x->ia', 'x->jb']),
    #Diagram('iax,jbx->iajb',          ('Rvv_X1_ov', 'Rov_X1X1_ov'),      -2.0, path=['x->ia', 'x->jb']),

    Diagram('iax,jbx->iajb',          ('Roo_X1_ov', 'Rov_part_oo'),      +1.0, path=['x->ia', 'x->jb']),
    #Diagram('iax,jbx->iajb',          ('Roo_X1_ov', 'Rov'),              -2.0, path=['x->ia', 'x->jb']),
    #Diagram('iax,jbx->iajb',          ('Roo_X1_ov', 'Roo_X1_ov'),        +1.0, path=['x->ia', 'x->jb']),
    #Diagram('iax,jbx->iajb',          ('Roo_X1_ov', 'Rov_X1X1_ov'),      +2.0, path=['x->ia', 'x->jb']),

    Diagram('iax,jbx->iajb',          ('Roo_X1_ov', 'Rvv_X1_ov'),        -2.0, path=['x->ia', 'x->jb']),
    Diagram('iax,jbx->iajb',          ('Rov_X1X1_ov', 'Rov_X1X1_ov'),    +1.0, path=['x->ia', 'x->jb']),
    Diagram('iax,jbx->iajb',          ('Rov_X1X1_ov', 'Rov'),            -2.0, path=['x->ia', 'x->jb']),
]

CCSD_X2_full_common = [
    Diagram('ki,kajb->iajb',          ('Loo', 'X2'),                     -2.0, path=['k->i', 'k->ajb']),
    Diagram('ac,icjb->iajb',          ('Lvv', 'X2'),                     -2.0, path=['c->a', 'c->ijb']),
    Diagram('kcia,kcjb->iajb',        ('Hovov', 'X2_symm'),              +2.0, path=['kc->ia', 'kc->jb']),
    Diagram('kcld,kcia,ldjb->iajb',   ('Hovov', 'X2_symm', 'X2_symm'),   +1.0, path=['kc->ld', 'kc->ia', 'ld->jb']), # HovovsameXsameXsame
    Diagram('kcld,ic,ka,ldjb->iajb',  ('Hovov', 'X1', 'X1', 'X2_symm'),  -2.0, path=['kc->ld', 'k->a', 'c->i', 'ld->jb']), # HovovsameXsameX1X1
    #Diagram('kcad,id,kcjb->iajb',     ('Hovvv', 'X1', 'X2_symm'),        +2.0, path=['kc->d', 'kcd->a', 'd->i', 'kc->jb']), # Hsame
    Diagram('kcli,la,kcjb->iajb',     ('Hovoo', 'X1', 'X2_symm'),        -2.0, path=['kc->l', 'kcl->i', 'l->a', 'kc->jb']), # Hsame

    Diagram('kcld,ldic,kajb->iajb',   ('Hovov', 'X2_symm', 'X2'),        -2.0, path=['k->cld', 'ldc->i', 'k->ajb']),  # H31_same o
    Diagram('kcld,ldka,icjb->iajb',   ('Hovov', 'X2_symm', 'X2'),        -2.0, path=['c->kld', 'ldk->a', 'c->ijb']),  # H31_same v
]

CCSD_X2_full_nonDF = [
    Diagram('kcda,id,kcjb->iajb',     ('Hovvv', 'X1', 'X2_symm'),        +2.0, path=['kc->d', 'kcd->a', 'd->i', 'kc->jb']), # Hsame
]

CCSD_X2_full_DF = [
    Diagram('kcx,iax,kcjb->iajb',     ('Rov', 'Rvv_X1_ov', 'X2_symm'),   +2.0, path=['kc->x', 'x->ia', 'kc->jb']), # Hsame
]

CCSD_X2_minimal_common = [
    Diagram('ki,kajb->iajb',          ('Loo_full', 'X2'),                -2.0),
    Diagram('ac,icjb->iajb',          ('Lvv_full', 'X2'),                -2.0),
    Diagram('iax,jbx->iajb',          ('Rov_full', 'XRov_symm'),        +1.0),
]

CCSD_X2_minimal_nonDF = []

CCSD_X2_minimal_DF = []


def screen(diagrams, nX_expected, check=True):
    screened_diagrams = []
    for diagram in diagrams:
        nX_in_diagram = ''.join(diagram.symbols).count('X')
        if check:
            assert nX_in_diagram >= nX_expected
        if nX_in_diagram == nX_expected:
            screened_diagrams.append(diagram)
    return screened_diagrams


def screen_nonlinear_intermediates(intermediates):
    return [(key, screen(diagrams, key.count('X'))) for key, diagrams in intermediates]


def screen_nonlinear_diagrams(diagrams):
    return screen(diagrams, 0, check=False) + screen(diagrams, 1, check=False)


def screen_unused_intermediates(intermediates, diagrams):
    symbols_diagrams = set(sum((diagram.symbols for diagram in diagrams), ()))
    while True:
        symbols = symbols_diagrams
        for _, intermediate_diagrams in intermediates:
            symbols = symbols.union(set(sum((diagram.symbols for diagram in intermediate_diagrams), ())))
        new_intermediates = [(key, intermediate_diagrams) for key, intermediate_diagrams in intermediates if key in symbols]
        if len(new_intermediates) == len(intermediates):
            break
        intermediates = new_intermediates
    return intermediates


def get_intermediate_tensors(minimal_stc, linear):
    if minimal_stc:
        result = intermediate_tensors + intermediate_tensors_minimal_add
    else:
        result = intermediate_tensors
    if linear:
        result = screen_nonlinear_intermediates(result)
    return result


def get_CCSD_X1(minimal_stc, df_virtual, linear):
    if minimal_stc:
        if df_virtual:
            result = CCSD_X1_common + CCSD_X1_common_DF + CCSD_X1_minimal_common + CCSD_X1_minimal_DF
        else:
            result = CCSD_X1_common + CCSD_X1_common_nonDF + CCSD_X1_minimal_common + CCSD_X1_minimal_nonDF
    else:
        if df_virtual:
            result = CCSD_X1_common + CCSD_X1_common_DF + CCSD_X1_full_common + CCSD_X1_full_DF
        else:
            result = CCSD_X1_common + CCSD_X1_common_nonDF + CCSD_X1_full_common + CCSD_X1_full_nonDF
    if linear:
        result = screen_nonlinear_diagrams(result)
    return result


def get_CCSD_X2(minimal_stc, df_virtual, linear):
    if minimal_stc:
        if df_virtual:
            result = CCSD_X2_common + CCSD_X2_common_DF + CCSD_X2_minimal_common + CCSD_X2_minimal_DF
        else:
            result = CCSD_X2_common + CCSD_X2_common_nonDF + CCSD_X2_minimal_common + CCSD_X2_minimal_nonDF
    else:
        if df_virtual:
            result = CCSD_X2_common + CCSD_X2_common_DF + CCSD_X2_full_common + CCSD_X2_full_DF
        else:
            result = CCSD_X2_common + CCSD_X2_common_nonDF + CCSD_X2_full_common + CCSD_X2_full_nonDF
    if linear:
        result = screen_nonlinear_diagrams(result)
    return result
