<?php

/*
 * SPDX-FileCopyrightText: 2026 Recognize contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */
declare(strict_types=1);
namespace OCA\Recognize\Command;

use OCP\IDBConnection;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputArgument;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;

final class ExportAdaptiveFaces extends Command {

	public function __construct(private IDBConnection $db) {
		parent::__construct();
	}

	protected function configure(): void {
		$this->setName('recognize:adaptive:export')
			->setDescription('Export one user\'s private face embeddings for offline experiments; does not modify assignments')
			->addArgument('user', InputArgument::REQUIRED, 'Nextcloud user ID')
			->addArgument('embedding-model', InputArgument::REQUIRED, 'Embedding model identifier, including weights revision and preprocessing');
	}

	protected function execute(InputInterface $input, OutputInterface $output): int {
		$user = (string)$input->getArgument('user');
		$model = trim((string)$input->getArgument('embedding-model'));
		if ($user === '' || $model === '') {
			throw new \InvalidArgumentException('User and embedding model must be nonempty');
		}
		$qb = $this->db->getQueryBuilder();
		$qb->select('d.id', 'd.user_id', 'd.file_id', 'd.cluster_id', 'd.threshold', 'd.x', 'd.y', 'd.width', 'd.height', 'd.face_vector', 'c.title')
			->from('recognize_face_detections', 'd')
			->leftJoin('d', 'recognize_face_clusters', 'c', $qb->expr()->andX(
				$qb->expr()->eq('c.id', 'd.cluster_id'),
				$qb->expr()->eq('c.user_id', 'd.user_id'),
			))
			->where($qb->expr()->eq('d.user_id', $qb->createNamedParameter($user)))
			->orderBy('d.id');
		$result = $qb->executeQuery();
		$rows = $result->fetchAll();
		$result->closeCursor();
		foreach ($rows as &$row) {
			foreach (['id', 'file_id', 'cluster_id'] as $field) {
				$row[$field] = $row[$field] === null ? null : (int)$row[$field];
			}
			foreach (['threshold', 'x', 'y', 'width', 'height'] as $field) {
				$row[$field] = (float)$row[$field];
			}
			$row['vector'] = json_decode($row['face_vector'], true, 512, JSON_THROW_ON_ERROR);
			$row['vector_hash'] = hash('sha256', $row['face_vector']);
			$row['content_key'] = 'file:' . $row['file_id'];
			unset($row['face_vector']);
		}
		unset($row);
		$output->writeln(json_encode([
			'schema' => 'recognize-adaptive-snapshot-v1',
			'embedding_model' => $model,
			'faces' => $rows,
		], JSON_THROW_ON_ERROR | JSON_UNESCAPED_UNICODE));
		return 0;
	}
}
