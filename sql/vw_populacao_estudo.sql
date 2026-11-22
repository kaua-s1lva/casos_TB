SELECT
    -- características individuais
    f.faixa_etaria,
    f.sexo,
    f.teste_hiv,
    f.alcool,
    f.tabagismo,
    f.drogas,
    f.escolaridade,
    f.raca,
    -- características do município de residência
    l.idhm,                  -- desenvolvimento humano (Atlas 2010)
    -- l.prop_pobreza,       -- alternativa ao IDHM (use um ou outro)
    l.indice_gini,           -- desigualdade de renda (Atlas 2010)
    l.faixa_populacao,       -- porte do município (MUNIC 2024)
    l.cobertura_aps_2021,    -- cobertura da Atenção Primária (%)
    l.regiao,                -- região geográfica
    -- desfecho
    f.situacao_encerramento
FROM fnotificacao f
LEFT JOIN dlocal l
       ON l.id_local = f.id_ocal_esidencia
WHERE 1=1
AND f.faixa_etaria != '<= 18 anos'
AND f.imigrante != 'Sim'
AND f.sem_teto != 'Sim'
AND f.profissional_saude != 'Sim'
AND f.encarcerado != 'Sim'
AND (f.gestante LIKE '%Não%' OR f.gestante = 'Ignorado')
AND f.forma_clinica = 'Pulmonar'
AND f.teste_sensibilidade NOT IN (
  '1',  -- Resistente somente à Isoniazida
  'Resistente somente à Rifampicina',
  'Resistente à Isoniazida e Rifampicina',
  'Resistente a outras drogas de 1ª linha'
)
AND f.tempo_entre_encerramento_e_inicio_tratamento IS NOT NULL
AND f.situacao_encerramento IN ('Cura', 'Abandono', 'Abandono Primário');